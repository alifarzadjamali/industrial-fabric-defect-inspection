"""U-Net decoder with a compact ImageNet-pretrained ResNet-18 encoder."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.upsample(inputs)
        return self.layers(torch.cat([inputs, skip], dim=1))


class ResNet18UNet(nn.Module):
    """One-channel binary U-Net using ResNet-18 feature stages."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = resnet18(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        stem = self.stem(inputs)
        level1 = self.encoder1(self.pool(stem))
        level2 = self.encoder2(level1)
        level3 = self.encoder3(level2)
        bottleneck = self.encoder4(level3)
        decoded = self.decoder4(bottleneck, level3)
        decoded = self.decoder3(decoded, level2)
        decoded = self.decoder2(decoded, level1)
        decoded = self.decoder1(decoded, stem)
        return self.final(decoded)
