"""Source-sized tiled inference for grayscale fabric images."""

from __future__ import annotations

import numpy as np
import torch
from cv2 import INTER_LINEAR, resize
from torch import nn

from fabric_inspection.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def _pad_patch(patch: np.ndarray, size: int) -> np.ndarray:
    vertical = size - patch.shape[0]
    horizontal = size - patch.shape[1]
    if vertical < 0 or horizontal < 0:
        raise ValueError(f"Patch {patch.shape} exceeds tile size {size}")
    if not vertical and not horizontal:
        return patch.copy()
    mode = "reflect" if min(patch.shape) > 1 else "edge"
    return np.pad(patch, ((0, vertical), (0, horizontal)), mode=mode)


def _normalise_patch(patch: np.ndarray, source_size: int, image_size: int) -> torch.Tensor:
    padded = _pad_patch(patch, source_size)
    if source_size != image_size:
        padded = resize(padded, (image_size, image_size), interpolation=INTER_LINEAR)
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
    tile_size: int | None = None,
    overlap: int = 0,
) -> np.ndarray:
    """Predict a full grayscale image with tiles, blending overlapping predictions."""

    if image.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image")
    if image_size <= 0:
        raise ValueError("Image size must be positive")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    height, width = image.shape
    tile_size = image_size if tile_size is None else tile_size
    if tile_size <= 0:
        raise ValueError("Tile size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("Overlap must be non-negative and smaller than tile size")
    stride = tile_size - overlap

    def starts(length: int) -> list[int]:
        values = list(range(0, max(length - tile_size, 0) + 1, stride))
        edge = max(length - tile_size, 0)
        if not values or values[-1] != edge:
            values.append(edge)
        return values

    coordinates: list[tuple[int, int, int, int]] = []
    patches: list[torch.Tensor] = []
    for y in starts(height):
        for x in starts(width):
            patch_height = min(tile_size, height - y)
            patch_width = min(tile_size, width - x)
            patch = image[y : y + patch_height, x : x + patch_width]
            patches.append(_normalise_patch(patch, tile_size, image_size))
            coordinates.append((x, y, patch_width, patch_height))

    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    if overlap:
        axis = np.maximum(np.hanning(tile_size).astype(np.float32), 0.05)
        window = np.outer(axis, axis)
    else:
        window = np.ones((tile_size, tile_size), dtype=np.float32)
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
                if tile_size != image_size:
                    predicted_patch = resize(
                        predicted_patch, (tile_size, tile_size), interpolation=INTER_LINEAR
                    )
                local_weight = window[:patch_height, :patch_width]
                probability_sum[y : y + patch_height, x : x + patch_width] += (
                    predicted_patch[:patch_height, :patch_width] * local_weight
                )
                weight_sum[y : y + patch_height, x : x + patch_width] += local_weight
    return probability_sum / np.maximum(weight_sum, 1e-8)
