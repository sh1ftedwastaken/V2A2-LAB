"""
LightUNet Training Script
==========================
Run from the PROJECT ROOT:
    python unet_code/train.py
    python unet_code/train.py --loss dice --epochs 50 --batch_size 8
    python unet_code/train.py --loss combo --lr 3e-4
    python unet_code/train.py --loss focal --gamma 2.0
    python unet_code/train.py --resume checkpoints/best.pth

Expects data/ folder produced by preprocess.py:
    data/images/   *.jpg   (320x240 RGB)
    data/masks/    *.png   (320x240 single-channel, pixel = class index)

Classes:
    0  background
    1  lane
    2  solid yellow line
    3  solid white line
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

# Class definitions

NUM_CLASSES = 5
CLASS_NAMES = [
    "background",
    "lane",
    "solid yellow line",
    "solid white line",
    "obstacle vehicle",
]

# Model

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
    """
    Input:  (N, 3, 320, 240)
    Output: (N, num_classes, 320, 240)
    """
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


# Loss functions

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets.clamp(0), self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        card  = (probs + one_hot).sum(dims)
        return 1.0 - ((2.0 * inter + self.smooth) / (card + self.smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class IoULoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets.clamp(0), self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        union = (probs + one_hot - probs * one_hot).sum(dims)
        return 1.0 - ((inter + self.smooth) / (union + self.smooth)).mean()


class CombinedLoss(nn.Module):
    """CrossEntropy + Dice - good default for segmentation."""
    def __init__(self, num_classes: int, w_ce: float = 0.5, w_dice: float = 0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.w_ce   = w_ce
        self.w_dice = w_dice

    def forward(self, logits, targets):
        return self.w_ce * self.ce(logits, targets) + self.w_dice * self.dice(logits, targets)


def build_loss(loss_name: str, gamma: float) -> nn.Module:
    return {
        "ce":    nn.CrossEntropyLoss(),
        "dice":  DiceLoss(NUM_CLASSES),
        "focal": FocalLoss(gamma=gamma),
        "iou":   IoULoss(NUM_CLASSES),
        "combo": CombinedLoss(NUM_CLASSES),
    }[loss_name]


# Metrics

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    preds, targets: (N, H, W) long tensors on any device.
    Returns per-class IoU and mIoU (NaN classes excluded from mean).
    """
    ious = []
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        union = tp + fp + fn
        ious.append(tp / union if union > 0 else float("nan"))
    valid = [v for v in ious if not np.isnan(v)]
    return {"per_class": ious, "miou": float(np.mean(valid)) if valid else 0.0}


# Dataset

class SegDataset(Dataset):
    """
    Loads (image, mask) pairs from data/images/ and data/masks/.
    Filenames must match (same stem, different extension).
    """
    IMG_EXT = {".jpg", ".jpeg", ".png"}

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
        raise FileNotFoundError(f"No image for stem: {stem}")

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img  = Image.open(self._find_image(stem)).convert("RGB")
        mask = Image.open(self.masks_dir / (stem + ".png"))

        if img.size != self.target_size:
            img  = img.resize(self.target_size, Image.BILINEAR)
        if mask.size != self.target_size:
            mask = mask.resize(self.target_size, Image.NEAREST)

        if self.augment and torch.rand(1).item() > 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return (
            self.to_tensor(img),
            torch.from_numpy(np.array(mask)).long()
        )


# Training helpers

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.autocast(device_type="cuda"):
                loss = criterion(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_miou   = []
    per_class_ious = [[] for _ in range(NUM_CLASSES)]

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item() * imgs.size(0)
        metrics = compute_metrics(logits.argmax(1), masks)
        all_miou.append(metrics["miou"])
        for c, v in enumerate(metrics["per_class"]):
            if not np.isnan(v):
                per_class_ious[c].append(v)

    avg_loss = total_loss / len(loader.dataset)
    avg_miou = float(np.mean(all_miou)) if all_miou else 0.0
    avg_per_class = [
        float(np.mean(v)) if v else float("nan")
        for v in per_class_ious
    ]
    return avg_loss, avg_miou, avg_per_class


# Main 

def parse_args():
    p = argparse.ArgumentParser(description="Train LightUNet")
    p.add_argument("--images_dir",  default="data/images")
    p.add_argument("--masks_dir",   default="data/masks")
    p.add_argument("--output_dir",  default="checkpoints")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--val_split",   type=float, default=0.15)
    p.add_argument("--dropout_p",   type=float, default=0.3)
    p.add_argument("--loss",        default="combo",
                   choices=["ce", "dice", "focal", "iou", "combo"])
    p.add_argument("--gamma",       type=float, default=2.0,
                   help="Focal loss gamma (ignored for other losses)")
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--resume",      default="")
    p.add_argument("--no_amp",      action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"[device]  {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Datasets
    full_ds = SegDataset(args.images_dir, args.masks_dir, augment=False)
    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
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
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(val_ds, val_idx),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    print(f"[data]    train={len(train_idx)}  val={len(val_idx)}")

    # Model
    model = LightUNet(num_classes=NUM_CLASSES, dropout_p=args.dropout_p).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model]   LightUNet  params={n_params:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = build_loss(args.loss, args.gamma).to(device)
    print(f"[loss]    {args.loss}" + (f"  gamma={args.gamma}" if args.loss == "focal" else ""))

    use_amp = not args.no_amp and device.type == "cuda"
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("[amp]     mixed precision ON")

    # Resume
    start_epoch = 0
    best_miou   = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        print(f"[resume]  epoch {start_epoch}  best_miou={best_miou:.4f}")

    # Training loop
    print(f"\n{'Ep':>4}  {'Train':>8}  {'Val':>8}  {'mIoU':>6}  "
          f"{'bg':>5} {'lane':>5} {'yel':>5} {'wht':>5} {'obs':>5}  {'t':>5}")
    print("-" * 75)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, miou, per_cls = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        pc = [f"{v:.3f}" if not np.isnan(v) else "  - " for v in per_cls]
        print(f"{epoch+1:>4}  {train_loss:>8.4f}  {val_loss:>8.4f}  {miou:>6.4f}  "
              f"{pc[0]:>5} {pc[1]:>5} {pc[2]:>5} {pc[3]:>5} {pc[4]:>5}  "
              f"{time.time()-t0:>4.1f}s")

        if miou > best_miou:
            best_miou = miou
            path = os.path.join(args.output_dir, "best.pth")
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_miou": best_miou, "args": vars(args),
            }, path)
            print(f"       Best saved  mIoU={best_miou:.4f}  -> {path}")

    torch.save({"epoch": args.epochs - 1, "model": model.state_dict(),
                "args": vars(args)},
               os.path.join(args.output_dir, "last.pth"))
    print(f"\n[done]  best mIoU={best_miou:.4f}")


if __name__ == "__main__":
    main()
