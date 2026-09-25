"""Patch dataset with paired, physically plausible image/mask augmentation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


@lru_cache(maxsize=24)
def _cached_grayscale(path: str) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    array.setflags(write=False)
    return array


def _parse_mask_paths(value: object) -> list[str]:
    return [item for item in str(value).split("|") if item and item != "nan"]


def _pad_patch(array: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
    vertical = size - array.shape[0]
    horizontal = size - array.shape[1]
    if vertical < 0 or horizontal < 0:
        raise ValueError(f"Patch {array.shape} exceeds configured input size {size}")
    if not vertical and not horizontal:
        return array.copy()
    mode = "constant" if is_mask else "reflect"
    return np.pad(array, ((0, vertical), (0, horizontal)), mode=mode)


def _augment(
    image: np.ndarray, mask: np.ndarray, profile: str = "standard"
) -> tuple[np.ndarray, np.ndarray]:
    if profile not in {"standard", "robust"}:
        raise ValueError(f"Unknown augmentation profile: {profile}")
    if np.random.random() < 0.5:
        image, mask = np.fliplr(image), np.fliplr(mask)
    if np.random.random() < 0.5:
        image, mask = np.flipud(image), np.flipud(mask)
    if np.random.random() < 0.7:
        limit = 10.0 if profile == "robust" else 7.0
        angle = float(np.random.uniform(-limit, limit))
        height, width = image.shape
        transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(
            image,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpAffine(
            mask.astype(np.uint8),
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(bool)
    image = image.astype(np.float32) / 255.0
    photometric_range = (0.85, 1.15) if profile == "robust" else (0.92, 1.08)
    brightness = float(np.random.uniform(*photometric_range))
    contrast = float(np.random.uniform(*photometric_range))
    image = (image - image.mean()) * contrast + image.mean()
    image = image * brightness
    if profile == "robust" and np.random.random() < 0.25:
        sigma = float(np.random.uniform(0.2, 1.2))
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    if np.random.random() < (0.4 if profile == "robust" else 0.25):
        noise_limit = 5.0 / 255.0 if profile == "robust" else 0.015
        image += np.random.normal(0.0, np.random.uniform(0.0, noise_limit), image.shape)
    return np.clip(image, 0.0, 1.0), mask


class AitexPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        image_size: int = 256,
        augment: bool = False,
        source_size: int | None = None,
        augmentation_profile: str = "standard",
    ) -> None:
        if image_size <= 0:
            raise ValueError("Image size must be positive")
        if source_size is not None and source_size <= 0:
            raise ValueError("Source size must be positive")
        if augmentation_profile not in {"standard", "robust"}:
            raise ValueError(f"Unknown augmentation profile: {augmentation_profile}")
        selected = manifest[
            (manifest["split"] == split) & manifest["has_segmentation_target"].astype(bool)
        ]
        self.rows = selected.reset_index(drop=True)
        self.image_size = image_size
        self.source_size = source_size or image_size
        self.augment = augment
        self.augmentation_profile = augmentation_profile
        if self.rows.empty:
            raise ValueError(f"No usable patches found for split {split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def positive_flags(self) -> np.ndarray:
        return self.rows["is_positive"].astype(bool).to_numpy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[index]
        x, y = int(row["x"]), int(row["y"])
        width, height = int(row["width"]), int(row["height"])
        image = _cached_grayscale(str(row["image_path"]))[y : y + height, x : x + width]
        mask = np.zeros((height, width), dtype=bool)
        for path in _parse_mask_paths(row["mask_paths"]):
            mask |= _cached_grayscale(path)[y : y + height, x : x + width] > 0
        image = _pad_patch(image, self.source_size)
        mask = _pad_patch(mask, self.source_size, is_mask=True)
        if self.source_size != self.image_size:
            image = cv2.resize(
                image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
            mask = cv2.resize(
                mask.astype(np.uint8),
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if self.augment:
            image, mask = _augment(image, mask, self.augmentation_profile)
        else:
            image = image.astype(np.float32) / 255.0
        channels = np.repeat(image[None, :, :], 3, axis=0)
        channels = (channels - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(np.ascontiguousarray(channels)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask[None, :, :])).float()
        return image_tensor, mask_tensor


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def make_dataloaders(
    manifest_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    positive_sampling_fraction: float,
    seed: int,
    source_size: int | None = None,
    augmentation_profile: str = "standard",
    small_defect_power: float = 0.0,
    hard_negative_fabrics: tuple[str, ...] = (),
    hard_negative_multiplier: float = 1.0,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    manifest = pd.read_csv(manifest_path)
    train_dataset = AitexPatchDataset(
        manifest,
        "train",
        image_size,
        augment=True,
        source_size=source_size,
        augmentation_profile=augmentation_profile,
    )
    validation_dataset = AitexPatchDataset(
        manifest, "validation", image_size, augment=False, source_size=source_size
    )
    flags = train_dataset.positive_flags
    positive_count = int(flags.sum())
    negative_count = int((~flags).sum())
    if not positive_count or not negative_count:
        raise ValueError("Balanced sampling requires positive and negative training patches")
    positive_weights = np.ones(len(train_dataset), dtype=np.float64)
    if small_defect_power > 0:
        pixels = train_dataset.rows["mask_pixels"].to_numpy(dtype=np.float64)
        positive_pixels = pixels[flags]
        positive_weights[flags] = (
            positive_pixels.max() / np.maximum(positive_pixels, 1.0)
        ) ** small_defect_power
    negative_weights = np.ones(len(train_dataset), dtype=np.float64)
    hard = train_dataset.rows["fabric_code"].astype(str).isin(hard_negative_fabrics).to_numpy()
    negative_weights[(~flags) & hard] = hard_negative_multiplier
    positive_weights /= positive_weights[flags].sum()
    negative_weights /= negative_weights[~flags].sum()
    weights = np.where(
        flags,
        positive_sampling_fraction * positive_weights,
        (1.0 - positive_sampling_fraction) * negative_weights,
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **common)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    counts = {
        "train_patches": len(train_dataset),
        "train_positive_patches": positive_count,
        "train_negative_patches": negative_count,
        "validation_patches": len(validation_dataset),
        "validation_positive_patches": int(validation_dataset.positive_flags.sum()),
    }
    return train_loader, validation_loader, counts
