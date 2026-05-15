"""PyTorch segmentation loss functions adapted from the original TensorFlow version.

These losses are written for multiclass segmentation models that return logits
with shape (N, C, H, W) and masks as class indices with shape (N, H, W).
"""
from __future__ import annotations #keep this since the limo car has a python version older
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.25
GAMMA = 2.0
EPSILON = 1e-6
SMOOTH = 1.0


def _prepare_class_weights(
    class_weights,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if class_weights is None:
        return None

    weights = torch.as_tensor(class_weights, device=device, dtype=dtype)
    if weights.numel() != num_classes:
        raise ValueError(
            f"Expected {num_classes} class weights, got {weights.numel()}."
        )
    return weights


def _weighted_class_average(
    values: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    if class_weights is None:
        return values.mean()

    weights = class_weights / class_weights.sum().clamp_min(EPSILON)
    return (values * weights).sum()


def _one_hot_targets(targets: torch.Tensor, num_classes: int, dtype: torch.dtype) -> torch.Tensor:
    if targets.ndim == 4:
        if targets.size(1) != num_classes:
            raise ValueError(
                f"Expected one-hot targets with {num_classes} channels, got {targets.size(1)}."
            )
        return targets.to(dtype=dtype)

    if targets.ndim != 3:
        raise ValueError(
            "Targets must have shape (N, H, W) with class indices or (N, C, H, W) as one-hot masks."
        )

    return F.one_hot(
        targets.long().clamp(min=0, max=num_classes - 1),
        num_classes=num_classes,
    ).permute(0, 3, 1, 2).to(dtype=dtype)


def _probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError("Logits must have shape (N, C, H, W).")
    # Compute custom losses in fp32 even when AMP is enabled to avoid NaNs.
    return F.softmax(logits.float(), dim=1)


def _dice_score_from_probs(probs: torch.Tensor, one_hot: torch.Tensor, smooth: float) -> torch.Tensor:
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dim=dims)
    denominator = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    return (2.0 * intersection + smooth) / (denominator + smooth)


def _jaccard_score_from_probs(probs: torch.Tensor, one_hot: torch.Tensor, smooth: float) -> torch.Tensor:
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dim=dims)
    union = (probs + one_hot - probs * one_hot).sum(dim=dims)
    return (intersection + smooth) / (union + smooth)


def _flatten_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if probs.ndim != 4:
        raise ValueError("Probabilities must have shape (N, C, H, W).")
    if targets.ndim != 3:
        raise ValueError("Targets must have shape (N, H, W).")

    probs = probs.permute(0, 2, 3, 1).reshape(-1, probs.size(1))
    targets = targets.reshape(-1)
    return probs, targets


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(dim=0)
    union = gts + (1.0 - gt_sorted).cumsum(dim=0)
    jaccard = 1.0 - intersection / union.clamp_min(EPSILON)
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = SMOOTH, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = _probabilities(logits)
        one_hot = _one_hot_targets(targets, self.num_classes, probs.dtype)
        class_weights = _prepare_class_weights(
            self.class_weights, self.num_classes, probs.device, probs.dtype
        )
        dice = _dice_score_from_probs(probs, one_hot, self.smooth)
        return 1.0 - _weighted_class_average(dice, class_weights)


class LogCoshDiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = SMOOTH, class_weights=None):
        super().__init__()
        self.dice = DiceLoss(
            num_classes=num_classes,
            smooth=smooth,
            class_weights=class_weights,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets).float()
        return dice_loss + F.softplus(-2.0 * dice_loss) - math.log(2.0)


class BCEDiceLoss(nn.Module):
    """Multiclass adaptation of BCE + Dice using softmax probabilities and one-hot masks."""

    def __init__(
        self,
        num_classes: int,
        smooth: float = SMOOTH,
        epsilon: float = EPSILON,
        class_weights=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.epsilon = epsilon
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = _probabilities(logits).clamp(min=self.epsilon, max=1.0 - self.epsilon)
        one_hot = _one_hot_targets(targets, self.num_classes, probs.dtype)
        class_weights = _prepare_class_weights(
            self.class_weights, self.num_classes, probs.device, probs.dtype
        )

        bce_per_class = F.binary_cross_entropy(
            probs, one_hot, reduction="none"
        ).mean(dim=(0, 2, 3))
        weighted_bce = _weighted_class_average(bce_per_class, class_weights)
        weighted_dice = 1.0 - _weighted_class_average(
            _dice_score_from_probs(probs, one_hot, self.smooth),
            class_weights,
        )
        return 0.5 * (weighted_bce + weighted_dice)


class JaccardLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = SMOOTH, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = _probabilities(logits)
        one_hot = _one_hot_targets(targets, self.num_classes, probs.dtype)
        class_weights = _prepare_class_weights(
            self.class_weights, self.num_classes, probs.device, probs.dtype
        )
        jaccard = _jaccard_score_from_probs(probs, one_hot, self.smooth)
        return 1.0 - _weighted_class_average(jaccard, class_weights)


class LovaszSoftmaxLoss(nn.Module):
    """Multiclass Lovasz-Softmax loss for direct IoU optimization."""

    def __init__(self, num_classes: int, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = _probabilities(logits)
        probs_flat, targets_flat = _flatten_probs(probs, targets.long())
        class_weights = _prepare_class_weights(
            self.class_weights, self.num_classes, probs.device, probs.dtype
        )

        class_losses = []
        class_ids = []
        for class_idx in range(self.num_classes):
            fg = (targets_flat == class_idx).to(dtype=probs.dtype)
            if fg.sum() == 0:
                continue

            errors = (fg - probs_flat[:, class_idx]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = _lovasz_grad(fg_sorted)
            class_losses.append(torch.dot(errors_sorted, grad))
            class_ids.append(class_idx)

        if not class_losses:
            return probs.sum() * 0.0

        losses = torch.stack(class_losses)
        if class_weights is None:
            return losses.mean()

        selected_weights = class_weights[torch.tensor(class_ids, device=probs.device)]
        return _weighted_class_average(losses, selected_weights)


class FocalLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        epsilon: float = EPSILON,
        class_weights=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = _probabilities(logits).clamp(min=self.epsilon, max=1.0 - self.epsilon)
        one_hot = _one_hot_targets(targets, self.num_classes, probs.dtype)
        class_weights = _prepare_class_weights(
            self.class_weights, self.num_classes, probs.device, probs.dtype
        )

        pos_loss = -self.alpha * one_hot * ((1.0 - probs) ** self.gamma) * torch.log(probs)
        neg_loss = -(1.0 - self.alpha) * (1.0 - one_hot) * (probs ** self.gamma) * torch.log(1.0 - probs)
        focal_per_class = (pos_loss + neg_loss).mean(dim=(0, 2, 3))
        return _weighted_class_average(focal_per_class, class_weights)


class SSIMLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        window_size: int = 11,
        max_val: float = 1.0,
        epsilon: float = EPSILON,
        class_weights=None,
    ):
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd for SSIMLoss.")

        self.num_classes = num_classes
        self.window_size = window_size
        self.max_val = max_val
        self.epsilon = epsilon
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = _probabilities(logits)
        one_hot = _one_hot_targets(targets, self.num_classes, probs.dtype)
        class_weights = _prepare_class_weights(
            self.class_weights, self.num_classes, probs.device, probs.dtype
        )
        pad = self.window_size // 2

        mu_x = F.avg_pool2d(probs, self.window_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(one_hot, self.window_size, stride=1, padding=pad)

        sigma_x = F.avg_pool2d(probs * probs, self.window_size, stride=1, padding=pad) - mu_x.pow(2)
        sigma_y = F.avg_pool2d(one_hot * one_hot, self.window_size, stride=1, padding=pad) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(probs * one_hot, self.window_size, stride=1, padding=pad) - mu_x * mu_y

        c1 = (0.01 * self.max_val) ** 2
        c2 = (0.03 * self.max_val) ** 2

        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
        ssim_map = numerator / (denominator + self.epsilon)
        ssim_per_class = ssim_map.mean(dim=(0, 2, 3))
        return 1.0 - _weighted_class_average(ssim_per_class, class_weights)


class UNet3PlusHybridLoss(nn.Module):
    """Multiclass adaptation of the UNet3+ hybrid loss."""

    def __init__(
        self,
        num_classes: int,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        smooth: float = SMOOTH,
        epsilon: float = EPSILON,
        class_weights=None,
    ):
        super().__init__()
        self.focal = FocalLoss(
            num_classes=num_classes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            class_weights=class_weights,
        )
        self.ssim = SSIMLoss(
            num_classes=num_classes,
            epsilon=epsilon,
            class_weights=class_weights,
        )
        self.jaccard = JaccardLoss(
            num_classes=num_classes,
            smooth=smooth,
            class_weights=class_weights,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, targets) + self.ssim(logits, targets) + self.jaccard(logits, targets)


class Semantic_loss_functions:
    """Compatibility wrapper that keeps the original method-style API."""

    def __init__(
        self,
        num_classes: int,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        smooth: float = SMOOTH,
        epsilon: float = EPSILON,
        class_weights=None,
    ):
        self.num_classes = num_classes
        self.dice_module = DiceLoss(
            num_classes=num_classes,
            smooth=smooth,
            class_weights=class_weights,
        )
        self.log_cosh_dice_module = LogCoshDiceLoss(
            num_classes=num_classes,
            smooth=smooth,
            class_weights=class_weights,
        )
        self.bce_dice_module = BCEDiceLoss(
            num_classes=num_classes,
            smooth=smooth,
            epsilon=epsilon,
            class_weights=class_weights,
        )
        self.jaccard_module = JaccardLoss(
            num_classes=num_classes,
            smooth=smooth,
            class_weights=class_weights,
        )
        self.unet3p_hybrid_module = UNet3PlusHybridLoss(
            num_classes=num_classes,
            alpha=alpha,
            gamma=gamma,
            smooth=smooth,
            epsilon=epsilon,
            class_weights=class_weights,
        )

    def generalized_dice_coefficient(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.dice_module(y_pred, y_true)

    def dice_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return self.dice_module(y_pred, y_true)

    def bce_dice_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return self.bce_dice_module(y_pred, y_true)

    def log_cosh_dice_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return self.log_cosh_dice_module(y_pred, y_true)

    def jacard_similarity(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        probs = _probabilities(y_pred)
        one_hot = _one_hot_targets(y_true, self.num_classes, probs.dtype)
        return _jaccard_score_from_probs(probs, one_hot, SMOOTH).mean()

    def jacard_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return self.jaccard_module(y_pred, y_true)

    def unet3p_hybrid_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return self.unet3p_hybrid_module(y_pred, y_true)
