"""Source-sized tiled inference for grayscale fabric images."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from cv2 import INTER_LINEAR, resize
from torch import nn

from fabric_inspection.data.dataset import _normalise_grayscale


def _pad_patch(patch: np.ndarray, size: int) -> np.ndarray:
    vertical = size - patch.shape[0]
    horizontal = size - patch.shape[1]
    if vertical < 0 or horizontal < 0:
        raise ValueError(f"Patch {patch.shape} exceeds tile size {size}")
    if not vertical and not horizontal:
        return patch
    mode = "reflect" if min(patch.shape) > 1 else "edge"
    return np.pad(patch, ((0, vertical), (0, horizontal)), mode=mode)


def _normalise_patch(patch: np.ndarray, source_size: int, image_size: int) -> torch.Tensor:
    padded = _pad_patch(patch, source_size)
    if source_size != image_size:
        padded = resize(padded, (image_size, image_size), interpolation=INTER_LINEAR)
    scaled = padded.astype(np.float32) / 255.0
    channels = _normalise_grayscale(scaled)
    # ``scaled`` is float32, so the normalized channels already have PyTorch's
    # default floating-point dtype.  Avoid an otherwise redundant tensor copy.
    return torch.from_numpy(np.ascontiguousarray(channels))


@lru_cache(maxsize=16)
def _blend_window(tile_size: int, overlap: int) -> np.ndarray:
    if overlap:
        axis = np.maximum(np.hanning(tile_size).astype(np.float32), 0.05)
        window = np.outer(axis, axis)
    else:
        window = np.ones((tile_size, tile_size), dtype=np.float32)
    window.setflags(write=False)
    return window


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
    if image.size == 0:
        raise ValueError("Image must contain at least one pixel")
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

    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    window = _blend_window(tile_size, overlap)
    was_training = model.training
    model.eval()
    patch_batch = torch.empty(
        (batch_size, 3, image_size, image_size),
        dtype=torch.float32,
        pin_memory=device.type == "cuda",
    )
    patch_count = 0
    coordinates: list[tuple[int, int, int, int]] = []

    def predict_batch() -> None:
        nonlocal patch_count
        if not patch_count:
            return
        batch = patch_batch[:patch_count].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision and device.type == "cuda",
        ):
            output = torch.sigmoid(model(batch)).float().cpu().numpy()[:, 0]
        for predicted_patch, (x, y, patch_width, patch_height) in zip(
            output, coordinates, strict=True
        ):
            if tile_size != image_size:
                predicted_patch = resize(
                    predicted_patch, (tile_size, tile_size), interpolation=INTER_LINEAR
                )
            local_weight = window[:patch_height, :patch_width]
            local_prediction = predicted_patch[:patch_height, :patch_width]
            np.multiply(local_prediction, local_weight, out=local_prediction)
            probability_sum[y : y + patch_height, x : x + patch_width] += local_prediction
            weight_sum[y : y + patch_height, x : x + patch_width] += local_weight
        patch_count = 0
        coordinates.clear()

    try:
        with torch.inference_mode():
            for y in starts(height):
                for x in starts(width):
                    patch_height = min(tile_size, height - y)
                    patch_width = min(tile_size, width - x)
                    patch = image[y : y + patch_height, x : x + patch_width]
                    patch_batch[patch_count].copy_(
                        _normalise_patch(patch, tile_size, image_size)
                    )
                    patch_count += 1
                    coordinates.append((x, y, patch_width, patch_height))
                    if patch_count == batch_size:
                        predict_batch()
            predict_batch()
    finally:
        model.train(was_training)
    np.divide(probability_sum, weight_sum, out=probability_sum)
    return probability_sum
