"""Losses for highly imbalanced binary segmentation."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("Dice smoothing must be finite and positive")
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        positive_pixel_weight: float = 25.0,
    ) -> None:
        super().__init__()
        if not all(math.isfinite(value) and value >= 0 for value in (bce_weight, dice_weight)):
            raise ValueError("Loss weights must be finite and non-negative")
        if bce_weight + dice_weight <= 0:
            raise ValueError("At least one loss weight must be positive")
        if not math.isfinite(positive_pixel_weight) or positive_pixel_weight <= 0:
            raise ValueError("Positive pixel weight must be finite and positive")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer("positive_pixel_weight", torch.tensor(positive_pixel_weight))
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.positive_pixel_weight
        )
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)


class FocalDiceLoss(nn.Module):
    """Focal binary cross-entropy plus soft Dice for sparse defects."""

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if not all(math.isfinite(value) and value >= 0 for value in (focal_weight, dice_weight)):
            raise ValueError("Loss weights must be finite and non-negative")
        if focal_weight + dice_weight <= 0:
            raise ValueError("At least one loss weight must be positive")
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Focal alpha must be between zero and one")
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("Focal gamma must be finite and non-negative")
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probability = torch.sigmoid(logits)
        probability_true = probability * targets + (1 - probability) * (1 - targets)
        alpha_true = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = (alpha_true * (1 - probability_true).pow(self.gamma) * bce).mean()
        return self.focal_weight * focal + self.dice_weight * self.dice(logits, targets)
