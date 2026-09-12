"""Losses for highly imbalanced binary segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
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
        if bce_weight + dice_weight <= 0:
            raise ValueError("At least one loss weight must be positive")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer("positive_pixel_weight", torch.tensor(positive_pixel_weight))
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.positive_pixel_weight
        )
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)
