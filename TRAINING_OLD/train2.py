"""
LightUNet Training Script — adapted for Windows + color-mask dataset
======================================================================
Run from your project folder:
    python train2.py
    python train2.py --loss combo --epochs 300
    python train2.py --loss focal --gamma 2.0
    python train2.py --resume checkpoints/best.pth

Expects:
    data/images/       raw camera photos (.png)
    data/masks_single/ single-channel masks (.png) — run convert_masks.py first!

Classes:
    0  background
    1  road
    2  white lane
    3  yellow lane
    4  obstacle vehicle
"""

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

from TRAINING.loss_functions import BCEDiceLoss, JaccardLoss, LogCoshDiceLoss, UNet3PlusHybridLoss

# ── Class definitions ──────────────────────────────────────────────────────────
NUM_CLASSES  = 5
CLASS_NAMES  = ["background", "road", "white lane", "yellow lane", "vehicle"]
CLASS_WEIGHTS = [0.4, 0.5, 8.0, 5.0, 6.0]
AMP_UNSAFE_LOSSES = {
    "log_cosh_dice",
    "bce_dice",
    "jaccard",
    "unet3p_hybrid",
    "ce_log_cosh_dice",
    "ce_jaccard",
    "ce_dice_jaccard",
}

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
        self.stem = DoubleConv(3, 16)
        self.enc1 = DoubleConv(16, 32)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.enc2 = DoubleConv(32, 64, dropout_p)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.enc3 = DoubleConv(64, 128, dropout_p)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.bottleneck = DoubleConv(128, 128, dropout_p)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(192, 64, dropout_p)
        self.up2  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(96, 32)
        self.up3  = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec3 = DoubleConv(48, 16)
        self.head = nn.Conv2d(16, num_classes, 1)

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


# ── Loss functions ─────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets.clamp(0), self.num_classes).permute(0,3,1,2).float()
        dims    = (0, 2, 3)
        inter   = (probs * one_hot).sum(dims)
        card    = (probs + one_hot).sum(dims)
        return 1.0 - ((2.0 * inter + self.smooth) / (card + self.smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class CombinedLoss(nn.Module):
    """CrossEntropy + Dice — best default for segmentation with class imbalance."""
    def __init__(self, num_classes: int, device):
        super().__init__()
        # Tuned weights based on your dataset distribution:
        # background=63%, road=34%, white lane=0.8%, yellow=1.9%, vehicle=rare
        weights = torch.tensor(CLASS_WEIGHTS).to(device)
        self.ce   = nn.CrossEntropyLoss(weight=weights)
        self.dice = DiceLoss(num_classes)

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


class CELogCoshDiceLoss(nn.Module):
    """Weighted CrossEntropy + LogCoshDice."""

    def __init__(self, num_classes: int, device):
        super().__init__()
        weights = torch.tensor(CLASS_WEIGHTS).to(device)
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.log_cosh_dice = LogCoshDiceLoss(num_classes, class_weights=None)

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.log_cosh_dice(logits, targets)


class CEJaccardLoss(nn.Module):
    """Weighted CrossEntropy + Jaccard."""

    def __init__(self, num_classes: int, device):
        super().__init__()
        weights = torch.tensor(CLASS_WEIGHTS).to(device)
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.jaccard = JaccardLoss(num_classes, class_weights=None)

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.jaccard(logits, targets)


class CEDiceJaccardLoss(nn.Module):
    """Weighted CrossEntropy + balanced Dice/Jaccard overlap."""

    def __init__(self, num_classes: int, device):
        super().__init__()
        weights = torch.tensor(CLASS_WEIGHTS).to(device)
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.dice = DiceLoss(num_classes)
        self.jaccard = JaccardLoss(num_classes, class_weights=None)

    def forward(self, logits, targets):
        overlap = 0.5 * self.dice(logits, targets) + 0.5 * self.jaccard(logits, targets)
        return 0.5 * self.ce(logits, targets) + 0.5 * overlap


def build_loss(loss_name: str, gamma: float, device) -> nn.Module:
    losses = {
        "ce":            nn.CrossEntropyLoss(),
        "dice":          DiceLoss(NUM_CLASSES),
        "focal":         FocalLoss(gamma=gamma),
        "combo":         CombinedLoss(NUM_CLASSES, device),
        "log_cosh_dice": LogCoshDiceLoss(NUM_CLASSES, class_weights=None),
        "bce_dice":      BCEDiceLoss(NUM_CLASSES, class_weights=None),
        "jaccard":       JaccardLoss(NUM_CLASSES, class_weights=None),
        "unet3p_hybrid": UNet3PlusHybridLoss(
            NUM_CLASSES,
            gamma=gamma,
            class_weights=CLASS_WEIGHTS,
        ),
        "ce_log_cosh_dice": CELogCoshDiceLoss(NUM_CLASSES, device),
        "ce_jaccard":       CEJaccardLoss(NUM_CLASSES, device),
        "ce_dice_jaccard":  CEDiceJaccardLoss(NUM_CLASSES, device),
    }
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Choose from {list(losses.keys())}")
    return losses[loss_name]


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
    """
    Loads (image, mask) pairs.
    - Images: data/images/  any of .png .jpg .jpeg
    - Masks:  data/masks_single/  single-channel .png (pixel value = class 0-4)
    """
    IMG_EXT = {".png", ".jpg", ".jpeg"}

    def __init__(self, images_dir: str, masks_dir: str,
                 target_size: tuple = (320, 240), augment: bool = False):
        self.images_dir  = Path(images_dir)
        self.masks_dir   = Path(masks_dir)
        self.target_size = target_size  # (W, H) for PIL
        self.augment     = augment

        self.stems = sorted(
            p.stem for p in self.images_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXT
        )
        assert len(self.stems) > 0, f"No images found in {images_dir}"

        # ImageNet normalization — standard practice, helps model learn faster
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
        raise FileNotFoundError(
            f"No image for stem '{stem}' in {self.images_dir}\n"
            f"Make sure your raw photos are in data/images/"
        )

    def _find_mask(self, stem: str) -> Path:
        p = self.masks_dir / (stem + ".png")
        if not p.exists():
            raise FileNotFoundError(
                f"No mask for stem '{stem}' in {self.masks_dir}\n"
                f"Did you run convert_masks.py first?"
            )
        return p

    def __getitem__(self, idx):
        stem = self.stems[idx]

        # Load and resize image
        img = Image.open(self._find_image(stem)).convert("RGB")
        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.BILINEAR)

        # Load and resize mask (NEAREST keeps class values exact)
        mask = Image.open(self._find_mask(stem))
        if mask.size != self.target_size:
            mask = mask.resize(self.target_size, Image.NEAREST)

        # Augmentation — random horizontal flip only (safe for road scenes)
        if self.augment and torch.rand(1).item() > 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Random brightness (image only, not mask)
        if self.augment and torch.rand(1).item() > 0.5:
            factor = 0.7 + torch.rand(1).item() * 0.6  # 0.7 to 1.3
            img = transforms.functional.adjust_brightness(img, factor)

        return (
            self.to_tensor(img),
            torch.from_numpy(np.array(mask)).long()
        )


# ── Training helpers ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for batch_idx, (imgs, masks) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.autocast(device_type="cuda"):
                logits = model(imgs)
            loss = criterion(logits, masks)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss detected at batch {batch_idx}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(imgs), masks)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss detected at batch {batch_idx}: {loss.item()}")
            loss.backward()
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
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits       = model(imgs)
        loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss detected.")
        total_loss  += loss.item() * imgs.size(0)
        metrics      = compute_metrics(logits.argmax(1), masks)
        all_miou.append(metrics["miou"])
        for c, v in enumerate(metrics["per_class"]):
            if not np.isnan(v):
                per_class_ious[c].append(v)

    avg_loss      = total_loss / len(loader.dataset)
    avg_miou      = float(np.mean(all_miou)) if all_miou else 0.0
    avg_per_class = [
        float(np.mean(v)) if v else float("nan")
        for v in per_class_ious
    ]
    return avg_loss, avg_miou, avg_per_class


# ── Main ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train LightUNet")
    p.add_argument("--images_dir",  default="data/images")
    p.add_argument("--masks_dir",   default="data/masks_single")
    p.add_argument("--output_dir",  default="checkpoints")
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--val_split",   type=float, default=0.15)
    p.add_argument("--dropout_p",   type=float, default=0.3)
    p.add_argument("--loss",        default="combo",
                   choices=["ce", "dice", "focal", "combo", "log_cosh_dice", "bce_dice", "jaccard", "unet3p_hybrid", "ce_log_cosh_dice", "ce_jaccard", "ce_dice_jaccard"])
    p.add_argument("--gamma",       type=float, default=2.0)
    p.add_argument("--resume",      default="")
    p.add_argument("--no_amp",      action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[device]  {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Build datasets
    full_ds = SegDataset(args.images_dir, args.masks_dir, augment=False)
    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val

    # Fixed seed so train/val split is always the same
    train_idx, val_idx = random_split(
        range(len(full_ds)), [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_idx = list(train_idx)
    val_idx   = list(val_idx)

    # Augmentation only on training set
    train_ds = SegDataset(args.images_dir, args.masks_dir, augment=True)
    val_ds   = SegDataset(args.images_dir, args.masks_dir, augment=False)

    # num_workers=0 required on Windows to avoid multiprocessing errors
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

    # Model
    model    = LightUNet(num_classes=NUM_CLASSES, dropout_p=args.dropout_p).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model]   LightUNet  params={n_params:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = build_loss(args.loss, args.gamma, device).to(device)
    print(f"[loss]    {args.loss}" + (f"  gamma={args.gamma}" if args.loss in {"focal", "unet3p_hybrid"} else ""))

    # Mixed precision (AMP) — faster training on your RTX 5070
    use_amp = not args.no_amp and device.type == "cuda"
    if use_amp and args.loss in AMP_UNSAFE_LOSSES:
        use_amp = False
        print(f"[amp]     disabled for {args.loss} to avoid NaNs with mixed precision")
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("[amp]     mixed precision ON — faster training!")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_miou   = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        print(f"[resume]  epoch {start_epoch}  best_miou={best_miou:.4f}")

    # Print header
    print(f"\n{'Ep':>4}  {'Train':>8}  {'Val':>8}  {'mIoU':>6}  "
          f"{'bg':>5} {'road':>5} {'wht':>5} {'yel':>5} {'veh':>5}  {'time':>5}")
    print("-" * 78)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss              = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, miou, per_cls = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        pc = [f"{v:.3f}" if not np.isnan(v) else "  -  " for v in per_cls]
        print(f"{epoch+1:>4}  {train_loss:>8.4f}  {val_loss:>8.4f}  {miou:>6.4f}  "
              f"{pc[0]:>5} {pc[1]:>5} {pc[2]:>5} {pc[3]:>5} {pc[4]:>5}  "
              f"{time.time()-t0:>4.1f}s")

        # Save best model (based on mIoU — better metric than loss)
        if miou > best_miou:
            best_miou = miou
            path = os.path.join(args.output_dir, "best.pth")
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_miou": best_miou,
                "args":      vars(args),
            }, path)
            print(f"       --> Best saved  mIoU={best_miou:.4f}  →  {path}")

    # Save final model
    torch.save(
        {"epoch": args.epochs - 1, "model": model.state_dict(), "args": vars(args)},
        os.path.join(args.output_dir, "last.pth")
    )
    print(f"\n[done]  best mIoU={best_miou:.4f}")
    print(f"        Models saved in: {args.output_dir}/")


if __name__ == "__main__":
    main()
