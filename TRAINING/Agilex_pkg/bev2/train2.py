"""
LightUNet Training Script
==========================
Run from your project folder:
    python train2.py                              # default: combo loss, 300 epochs
    python train2.py --loss ce_jaccard            # BEST so far (0.9274 mIoU)
    python train2.py --loss ce_jaccard --epochs 400
    python train2.py --loss combo --epochs 300
    python train2.py --loss unet3p_hybrid --epochs 150
    python train2.py --resume checkpoints/best_lightunet_ce_jaccard.pth --loss ce_jaccard

Expects:
    data/images/       raw camera photos (.png or .jpg)
    data/masks_single/ single-channel masks (.png) — run convert_masks.py first!

Classes:
    0  background
    1  road
    2  white lane
    3  yellow lane
    4  obstacle vehicle
"""
from __future__ import annotations #keep this since the limo car has a python version older
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from loss_functions import BCEDiceLoss, JaccardLoss, LogCoshDiceLoss, LovaszSoftmaxLoss, UNet3PlusHybridLoss

# ── Class definitions ──────────────────────────────────────────────────────────
NUM_CLASSES = 5
CLASS_NAMES = ["background", "road", "white lane", "yellow lane", "vehicle"]
CLASS_WEIGHTS = [0.5, 2.0, 3.5, 3.5, 2.5]
# regular [0.4, 0.5, 12.0, 8.0, 7.0]
# normalized [0.07, 0.09, 2.15, 1.43, 1.25]
# [0.5, 2.0, 3.5, 3.5, 2.5]

AMP_UNSAFE_LOSSES = {
    "log_cosh_dice",
    "bce_dice",
    "jaccard",
    "unet3p_hybrid",
    "ce_log_cosh_dice",
    "ce_jaccard_lovasz",
    "ce_jaccard_logcosh",
}
LOSS_CHOICES = [
    "ce", "dice", "focal", "combo",
    "ce_jaccard",
    "ce_lovasz",
    "ce_log_cosh_dice",
    "ce_jaccard_lovasz",
    "ce_jaccard_logcosh",
    "lovasz_focal",
    "log_cosh_dice",
    "bce_dice",
    "jaccard",
    "unet3p_hybrid",
    "last_try"
]
MODEL_CHOICES = [
    "lightunet",
    "resnet34_unet",
]


def build_run_tag(args) -> str:
    parts = [args.model, args.loss]
    if args.switch_loss_to:
        parts.extend(["to", args.switch_loss_to])
    return "_".join(parts)


# ── Model ──────────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LightUNet(nn.Module):
    """Input: (N, 3, 240, 320) — Output: (N, num_classes, 240, 320)"""
    def __init__(self, num_classes: int = NUM_CLASSES, dropout_p: float = 0.3):
        super().__init__()
        self.stem       = DoubleConv(3, 16)
        self.enc1       = DoubleConv(16, 32)
        self.pool1      = nn.AvgPool2d(2, 2)
        self.enc2       = DoubleConv(32, 64, dropout_p)
        self.pool2      = nn.AvgPool2d(2, 2)
        self.enc3       = DoubleConv(64, 128, dropout_p)
        self.pool3      = nn.AvgPool2d(2, 2)
        self.bottleneck = DoubleConv(128, 128, dropout_p)
        self.up1        = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1       = DoubleConv(192, 64, dropout_p)
        self.up2        = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2       = DoubleConv(96, 32)
        self.up3        = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec3       = DoubleConv(48, 16)
        self.head       = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x0    = self.stem(x)
        skip1 = self.enc1(x0);  x = self.pool1(skip1)
        skip2 = self.enc2(x);   x = self.pool2(skip2)
        skip3 = self.enc3(x);   x = self.pool3(skip3)
        x = self.bottleneck(x)
        x = self.up1(x);  x = torch.cat([x, skip3], 1); x = self.dec1(x)
        x = self.up2(x);  x = torch.cat([x, skip2], 1); x = self.dec2(x)
        x = self.up3(x);  x = torch.cat([x, skip1], 1); x = self.dec3(x)
        return self.head(x)


def build_model(model_name: str, dropout_p: float, encoder_weights: str | None):
    if model_name == "lightunet":
        return LightUNet(num_classes=NUM_CLASSES, dropout_p=dropout_p)

    if model_name == "resnet34_unet":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "Model 'resnet34_unet' requires segmentation_models_pytorch. "
                "Install it with: pip install segmentation-models-pytorch"
            ) from exc

        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None,
        )

    raise ValueError(f"Unknown model: {model_name}")


# ── Loss functions ─────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = F.softmax(logits.float(), dim=1)   # fp32 for stability
        one_hot = F.one_hot(targets.clamp(0), self.num_classes).permute(0, 3, 1, 2).float()
        dims    = (0, 2, 3)
        inter   = (probs * one_hot).sum(dims)
        card    = (probs + one_hot).sum(dims)
        return 1.0 - ((2.0 * inter + self.smooth) / (card + self.smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits.float(), targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class CombinedLoss(nn.Module):
    """Weighted CrossEntropy + Dice. Your first strong baseline."""
    def __init__(self, num_classes: int, device):
        super().__init__()
        weights   = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
        self.ce   = nn.CrossEntropyLoss(weight=weights)
        self.dice = DiceLoss(num_classes)

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits.float(), targets) + 0.5 * self.dice(logits, targets)


class CEJaccardLoss(nn.Module):
    """Weighted CrossEntropy + Jaccard.
    BEST result so far (0.9274). CE handles class imbalance,
    Jaccard directly optimizes the IoU metric you're measuring.
    """
    def __init__(self, num_classes: int, device):
        super().__init__()
        weights      = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
        self.ce      = nn.CrossEntropyLoss(weight=weights)
        self.jaccard = JaccardLoss(num_classes, class_weights=None)

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits.float(), targets) + 0.5 * self.jaccard(logits, targets)


class CELovaszLoss(nn.Module):
    """Weighted CrossEntropy + Lovasz-Softmax."""

    def __init__(self, num_classes: int, device):
        super().__init__()
        weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.lovasz = LovaszSoftmaxLoss(num_classes, class_weights=None)

    def forward(self, logits, targets):
        return 0.6 * self.ce(logits.float(), targets) + 0.4 * self.lovasz(logits, targets)


class CEJaccardLovaszLoss(nn.Module):
    """Weighted CrossEntropy + Jaccard + Lovasz-Softmax."""

    def __init__(self, num_classes: int, device):
        super().__init__()
        weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.jaccard = JaccardLoss(num_classes, class_weights=None)
        self.lovasz = LovaszSoftmaxLoss(num_classes, class_weights=None)

    def forward(self, logits, targets):
        ce_w, jaccard_w, lovasz_w = (0.60, 0.25, 0.15)
        return (
            ce_w * self.ce(logits.float(), targets)
          + jaccard_w * self.jaccard(logits, targets)
          + lovasz_w * self.lovasz(logits, targets)
        )


class CELogCoshDiceLoss(nn.Module):
    """Weighted CrossEntropy + LogCosh Dice.
    LogCosh Dice is smoother than regular Dice on thin structures (white lane).
    """
    def __init__(self, num_classes: int, device):
        super().__init__()
        weights           = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
        self.ce           = nn.CrossEntropyLoss(weight=weights)
        self.log_cosh_dice = LogCoshDiceLoss(num_classes, class_weights=None)

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits.float(), targets) + 0.5 * self.log_cosh_dice(logits, targets)


class CEJaccardLogCoshLoss(nn.Module):
    """Weighted CE + Jaccard + LogCosh Dice — NEW combination to try.
    Rationale: CE fixes imbalance, Jaccard optimizes IoU directly,
    LogCosh Dice adds smoothness for thin structures like white lane.
    Each component contributes something different.
    """
    def __init__(self, num_classes: int, device):
        super().__init__()
        weights            = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
        self.ce            = nn.CrossEntropyLoss(weight=weights)
        self.jaccard       = JaccardLoss(num_classes, class_weights=None)
        self.log_cosh_dice = LogCoshDiceLoss(num_classes, class_weights=None)

    def forward(self, logits, targets):
        ce_w, jaccard_w, logcosh_w = (0.6, 0.25, 0.15)
        return (
            ce_w * self.ce(logits.float(), targets)
          + jaccard_w * self.jaccard(logits, targets)
          + logcosh_w * self.log_cosh_dice(logits, targets)
        )
        
class LovaszFocalLoss(nn.Module):
    def __init__(self, num_classes: int, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.lovasz = LovaszSoftmaxLoss(num_classes, class_weights=None)
        self.focal = FocalLoss(alpha=CLASS_WEIGHTS, gamma=self.gamma) 

    def forward(self, logits, targets):
        return 0.5 * self.focal(logits, targets) + 0.5 * self.lovasz(logits, targets)


class TopKCELoss(nn.Module):
    def __init__(self, k=0.1, weight=None):
        super().__init__()
        self.k = k
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.weight)
        view = ce.view(-1)
        num_pixels = view.numel()
        top_k_pixels = int(self.k * num_pixels)
        if top_k_pixels > 0:
            view, _ = torch.topk(view, top_k_pixels)
        return view.mean()


class CombinedPolishingLoss(nn.Module):
    def __init__(self, num_classes, weight=None, k=0.1):
        super().__init__()
        if weight is not None:
            self.register_buffer('weight', torch.tensor(weight, dtype=torch.float32))
        else:
            self.weight = None
            
        self.k = k
        self.dice = DiceLoss(num_classes)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits.float(), targets, reduction='none', weight=self.weight)
        view = ce.view(-1)
        num_pixels = view.numel()
        top_k_pixels = int(self.k * num_pixels)
        
        if top_k_pixels > 0:
            view, _ = torch.topk(view, top_k_pixels)
        
        tk_loss = view.mean()
        d_loss = self.dice(logits, targets)

        return (0.7 * tk_loss) + (0.3 * d_loss)
    
    
# ─────────────────────────────────────────────────────────────────────────────────────────

def build_loss(loss_name: str, gamma: float, device) -> nn.Module:
    losses = {
        "ce":                   nn.CrossEntropyLoss(),
        "dice":                 DiceLoss(NUM_CLASSES),
        "focal":                FocalLoss(gamma=gamma),
        "combo":                CombinedLoss(NUM_CLASSES, device),
        "ce_jaccard":           CEJaccardLoss(NUM_CLASSES, device),
        "ce_lovasz":            CELovaszLoss(NUM_CLASSES, device),
        "ce_log_cosh_dice":     CELogCoshDiceLoss(NUM_CLASSES, device),
        "ce_jaccard_lovasz":    CEJaccardLovaszLoss(NUM_CLASSES, device),
        "ce_jaccard_logcosh":   CEJaccardLogCoshLoss(NUM_CLASSES, device),  # NEW
        "lovasz_focal":         LovaszFocalLoss(NUM_CLASSES, gamma=2.0), 
        "log_cosh_dice":        LogCoshDiceLoss(NUM_CLASSES, class_weights=None),
        "bce_dice":             BCEDiceLoss(NUM_CLASSES, class_weights=None),
        "jaccard":              JaccardLoss(NUM_CLASSES, class_weights=None),
        "unet3p_hybrid":        UNet3PlusHybridLoss(NUM_CLASSES, gamma=gamma, class_weights=CLASS_WEIGHTS),
        "last_try":             CombinedPolishingLoss(NUM_CLASSES, CLASS_WEIGHTS, k=0.1)
    }
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Choose from: {list(losses.keys())}")
    return losses[loss_name]


def build_amp_state(loss_name: str, device: torch.device, no_amp: bool):
    use_amp = not no_amp and device.type == "cuda"
    amp_dtype = None
    scaler = None

    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if amp_dtype == torch.float16 and loss_name in AMP_UNSAFE_LOSSES:
            use_amp = False
            amp_dtype = None
            print(f"[amp]     disabled for {loss_name} because fp16 is unstable for this loss")
        elif amp_dtype == torch.float16:
            scaler = torch.amp.GradScaler("cuda")
            print("[amp]     mixed precision ON (float16)")
        else:
            print("[amp]     mixed precision ON (bfloat16)")

    return use_amp, amp_dtype, scaler


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Per-class IoU and mIoU. NaN = class not present in this batch."""
    ious = []
    for c in range(NUM_CLASSES):
        tp    = ((preds == c) & (targets == c)).sum().item()
        fp    = ((preds == c) & (targets != c)).sum().item()
        fn    = ((preds != c) & (targets == c)).sum().item()
        union = tp + fp + fn
        ious.append(tp / union if union > 0 else float("nan"))
    valid = [v for v in ious if not np.isnan(v)]
    return {"per_class": ious, "miou": float(np.mean(valid)) if valid else 0.0}


# ── Dataset ────────────────────────────────────────────────────────────────────
class SegDataset(Dataset):
    IMG_EXT = {".png", ".jpg", ".jpeg"}

    def __init__(self, images_dir: str, masks_dir: str,
                 target_size: tuple = (320, 240), augment: bool = False):
        self.images_dir  = Path(images_dir)
        self.masks_dir   = Path(masks_dir)
        self.target_size = target_size
        self.augment     = augment

        self.stems = sorted(
            p.stem for p in self.images_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXT
        )
        assert len(self.stems) > 0, f"No images found in {images_dir}"

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.stems)

    def _find_image(self, stem: str) -> Path:
        for ext in self.IMG_EXT:
            p = self.images_dir / (stem + ext)
            if p.exists():
                return p
        raise FileNotFoundError(f"No image for stem '{stem}' in {self.images_dir}")

    def _find_mask(self, stem: str) -> Path:
        p = self.masks_dir / (stem + ".png")
        if not p.exists():
            raise FileNotFoundError(f"No mask for stem '{stem}' in {self.masks_dir}")
        return p

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img  = Image.open(self._find_image(stem)).convert("RGB")
        mask = Image.open(self._find_mask(stem))

        if img.size  != self.target_size:
            img  = img.resize(self.target_size, Image.BILINEAR)
        if mask.size != self.target_size:
            mask = mask.resize(self.target_size, Image.NEAREST)

        # FIX 3: Expanded augmentation for better generalization
        if self.augment:
            # Horizontal flip (most useful for road scenes)
            if torch.rand(1).item() > 0.5:
                img  = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # Random brightness — helps with lighting variation between sessions
            if torch.rand(1).item() > 0.5:
                factor = 0.7 + torch.rand(1).item() * 0.6   # 0.7 to 1.3
                img = transforms.functional.adjust_brightness(img, factor)

            # NEW: Random contrast — helps with road/background confusion at corners
            if torch.rand(1).item() > 0.5:
                factor = 0.8 + torch.rand(1).item() * 0.4   # 0.8 to 1.2
                img = transforms.functional.adjust_contrast(img, factor)

            # NEW: Random color jitter — helps model not overfit to exact colors
            if torch.rand(1).item() > 0.7:
                factor = 0.9 + torch.rand(1).item() * 0.2   # 0.9 to 1.1
                img = transforms.functional.adjust_saturation(img, factor)

        return (
            self.to_tensor(img),
            torch.from_numpy(np.array(mask)).long()
        )


# ── Training helpers ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp=False, amp_dtype=torch.float16):
    model.train()
    total_loss = 0.0
    for batch_idx, (imgs, masks) in enumerate(loader, start=1):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(imgs)
            loss = criterion(logits, masks)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at batch {batch_idx}: {loss.item()}")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            logits = model(imgs)
            loss   = criterion(logits, masks)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at batch {batch_idx}: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss     = 0.0
    all_miou       = []
    per_class_ious = [[] for _ in range(NUM_CLASSES)]

    for imgs, masks in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss detected.")
        total_loss += loss.item() * imgs.size(0)
        metrics     = compute_metrics(logits.argmax(1), masks)
        all_miou.append(metrics["miou"])
        for c, v in enumerate(metrics["per_class"]):
            if not np.isnan(v):
                per_class_ious[c].append(v)

    return (
        total_loss / len(loader.dataset),
        float(np.mean(all_miou)) if all_miou else 0.0,
        [float(np.mean(v)) if v else float("nan") for v in per_class_ious],
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train segmentation model")
    p.add_argument("--images_dir",  default="data/images")
    p.add_argument("--masks_dir",   default="data/masks_single")
    p.add_argument("--output_dir",  default="checkpoints")
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--val_split",   type=float, default=0.15)
    p.add_argument("--dropout_p",   type=float, default=0.3)
    p.add_argument("--loss",        default="ce_jaccard",
                   choices=LOSS_CHOICES)
    p.add_argument("--model",       default="lightunet",
                   choices=MODEL_CHOICES)
    p.add_argument("--encoder_weights", default="imagenet",
                   choices=["imagenet", "none"],
                   help="Used only for SMP backbones like resnet34_unet.")
    p.add_argument("--gamma",       type=float, default=2.0)
    p.add_argument("--resume",      default="")
    p.add_argument("--no_amp",      action="store_true")
    p.add_argument("--switch_loss_to", default="")
    p.add_argument("--switch_loss_at", type=int, default=0)
    p.add_argument("--switch_lr_factor", type=float, default=1.0)
    p.add_argument("--switch_lr", type=float, default=None,
                   help="Optional absolute learning rate to set when the loss switches.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_weights = None if args.encoder_weights == "none" else args.encoder_weights
    run_tag = build_run_tag(args)
    best_path = os.path.join(args.output_dir, f"best_{run_tag}.pth")
    last_path = os.path.join(args.output_dir, f"last_{run_tag}.pth")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[device]  {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    full_ds   = SegDataset(args.images_dir, args.masks_dir, augment=False)
    n_val     = max(1, int(len(full_ds) * args.val_split))
    n_train   = len(full_ds) - n_val
    train_idx, val_idx = random_split(
        range(len(full_ds)), [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_idx = list(train_idx)
    val_idx   = list(val_idx)

    train_ds = SegDataset(args.images_dir, args.masks_dir, augment=True)
    val_ds   = SegDataset(args.images_dir, args.masks_dir, augment=False)

    train_loader = DataLoader(
        torch.utils.data.Subset(train_ds, train_idx),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(val_ds, val_idx),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    print(f"[data]    train={len(train_idx)}  val={len(val_idx)}")
    print(f"[save]    best={os.path.basename(best_path)}  last={os.path.basename(last_path)}")

    # Model
    model    = build_model(args.model, args.dropout_p, encoder_weights).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model]   {args.model}  params={n_params:,}")
    if args.model == "resnet34_unet":
        print(f"[encoder] {args.encoder_weights}")
    print(f"[weights] bg={CLASS_WEIGHTS[0]} road={CLASS_WEIGHTS[1]} "
          f"wht={CLASS_WEIGHTS[2]} yel={CLASS_WEIGHTS[3]} veh={CLASS_WEIGHTS[4]}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    current_loss_name = args.loss
    if args.switch_loss_to and args.switch_loss_at <= 0:
        raise ValueError("--switch_loss_at must be > 0 when --switch_loss_to is provided.")

    if args.switch_loss_to and args.switch_loss_to not in LOSS_CHOICES:
        raise ValueError(f"Unknown switch loss: {args.switch_loss_to}")

    if args.switch_lr is not None and args.switch_lr <= 0:
        raise ValueError("--switch_lr must be > 0 when provided.")

    criterion = build_loss(current_loss_name, args.gamma, device).to(device)
    print(f"[loss]    {current_loss_name}" + (f"  gamma={args.gamma}" if current_loss_name in {"focal", "unet3p_hybrid"} else ""))
    if args.switch_loss_to:
        if args.switch_lr is not None:
            print(f"[switch]  epoch {args.switch_loss_at} -> {args.switch_loss_to}  lr={args.switch_lr:.2e}")
        else:
            print(f"[switch]  epoch {args.switch_loss_at} -> {args.switch_loss_to}  lr x{args.switch_lr_factor}")

    use_amp, amp_dtype, scaler = build_amp_state(current_loss_name, device, args.no_amp)

    # Resume
    start_epoch = 0
    best_miou   = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt        = torch.load(args.resume, map_location=device)
        try:
            model.load_state_dict(ckpt["model"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint '{args.resume}' is not compatible with model '{args.model}'. "
                "Use a checkpoint from the same architecture or start without --resume."
            ) from exc
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            print("[resume]  checkpoint has no scheduler state; LR schedule restarts from current settings")
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        print(f"[resume]  epoch {start_epoch}  best_miou={best_miou:.4f}")

    if args.switch_loss_to and start_epoch + 1 >= args.switch_loss_at and current_loss_name != args.switch_loss_to:
        current_loss_name = args.switch_loss_to
        criterion = build_loss(current_loss_name, args.gamma, device).to(device)
        use_amp, amp_dtype, scaler = build_amp_state(current_loss_name, device, args.no_amp)
        print(f"[switch]  resume continues with {current_loss_name}")

    print(f"\n{'Ep':>4}  {'Train':>8}  {'Val':>8}  {'mIoU':>6}  "
          f"{'bg':>5} {'road':>5} {'wht':>5} {'yel':>5} {'veh':>5}  {'time':>5}")
    print("-" * 78)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        epoch_num = epoch + 1

        if args.switch_loss_to and epoch_num == args.switch_loss_at and current_loss_name != args.switch_loss_to:
            current_loss_name = args.switch_loss_to
            criterion = build_loss(current_loss_name, args.gamma, device).to(device)
            for param_group in optimizer.param_groups:
                if args.switch_lr is not None:
                    param_group["lr"] = args.switch_lr
                else:
                    param_group["lr"] *= args.switch_lr_factor
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - epoch),
                eta_min=1e-6,
            )
            use_amp, amp_dtype, scaler = build_amp_state(current_loss_name, device, args.no_amp)
            print(
                f"[switch]  epoch {epoch_num} -> {current_loss_name}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        train_loss              = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            use_amp=use_amp, amp_dtype=amp_dtype or torch.float16
        )
        val_loss, miou, per_cls = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        pc = [f"{v:.3f}" if not np.isnan(v) else "  -  " for v in per_cls]
        print(f"{epoch+1:>4}  {train_loss:>8.4f}  {val_loss:>8.4f}  {miou:>6.4f}  "
              f"{pc[0]:>5} {pc[1]:>5} {pc[2]:>5} {pc[3]:>5} {pc[4]:>5}  "
              f"{time.time()-t0:>4.1f}s")

        if miou > best_miou:
            best_miou = miou
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_miou": best_miou,
                "current_loss_name": current_loss_name,
                "args":      vars(args),
            }, best_path)
            print(f"       --> Best saved  mIoU={best_miou:.4f}  →  {best_path}")

    torch.save(
        {
            "epoch": args.epochs - 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": best_miou,
            "current_loss_name": current_loss_name,
            "args": vars(args),
        },
        last_path
    )
    print(f"\n[done]  best mIoU={best_miou:.4f}")
    print(f"        Models saved in: {args.output_dir}/")


if __name__ == "__main__":
    main()
