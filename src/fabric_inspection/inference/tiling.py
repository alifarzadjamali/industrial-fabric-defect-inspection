"""Source-sized tiled inference for grayscale fabric images."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fabric_inspection.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def _normalise_patch(patch: np.ndarray, image_size: int) -> torch.Tensor:
    height, width = patch.shape
    padded = np.pad(
        patch,
        ((0, image_size - height), (0, image_size - width)),
        mode="reflect",
    )
    scaled = padded.astype(np.float32) / 255.0
    channels = np.repeat(scaled[None, :, :], 3, axis=0)
    channels = (channels - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(channels)).float()


def predict_grayscale_image(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    image_size: int = 256,
    batch_size: int = 24,
    mixed_precision: bool = True,
) -> np.ndarray:
    """Predict a probability map with non-overlapping training-consistent tiles."""

    if image.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image")
    height, width = image.shape
    coordinates: list[tuple[int, int, int, int]] = []
    patches: list[torch.Tensor] = []
    for y in range(0, height, image_size):
        for x in range(0, width, image_size):
            patch_height = min(image_size, height - y)
            patch_width = min(image_size, width - x)
            patch = image[y : y + patch_height, x : x + patch_width]
            patches.append(_normalise_patch(patch, image_size))
            coordinates.append((x, y, patch_width, patch_height))

    probability = np.zeros((height, width), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            batch = torch.stack(patches[start : start + batch_size]).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                output = torch.sigmoid(model(batch)).float().cpu().numpy()[:, 0]
            for local_index, predicted_patch in enumerate(output):
                x, y, patch_width, patch_height = coordinates[start + local_index]
                probability[y : y + patch_height, x : x + patch_width] = predicted_patch[
                    :patch_height, :patch_width
                ]
    return probability
