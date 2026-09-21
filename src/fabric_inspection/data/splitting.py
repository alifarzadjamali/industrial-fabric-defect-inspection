"""Deterministic, original-image-level dataset splitting and patch manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from fabric_inspection.data.aitex import AitexRecord


def _stratification_key(frame: pd.DataFrame) -> pd.Series:
    detailed = frame["fabric_code"].astype(str) + "_" + frame["is_defective"].astype(str)
    if detailed.value_counts().min() >= 3:
        return detailed
    return frame["is_defective"].astype(str)


def create_split_manifest(
    records: list[AitexRecord],
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> pd.DataFrame:
    """Assign each source image to one split so its patches stay together."""

    if not records:
        raise ValueError("At least one source image is required to create splits")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test set")

    frame = pd.DataFrame(
        {
            "image_id": record.image_id,
            "image_path": str(record.image_path),
            "is_defective": record.is_defective,
            "defect_code": record.defect_code,
            "defect_name": record.defect_name,
            "fabric_code": record.fabric_code,
            "mask_paths": "|".join(str(path.resolve()) for path in record.mask_paths),
        }
        for record in records
    ).sort_values("image_id", ignore_index=True)
    if frame["image_id"].duplicated().any():
        duplicates = frame.loc[frame["image_id"].duplicated(), "image_id"].tolist()
        raise ValueError(f"Duplicate source image IDs: {duplicates}")

    train, remainder = train_test_split(
        frame,
        train_size=train_fraction,
        random_state=seed,
        stratify=_stratification_key(frame),
    )
    relative_validation = validation_fraction / (1.0 - train_fraction)
    validation, test = train_test_split(
        remainder,
        train_size=relative_validation,
        random_state=seed,
        stratify=_stratification_key(remainder),
    )
    assignments = []
    for name, subset in (("train", train), ("validation", validation), ("test", test)):
        assigned = subset.copy()
        assigned["split"] = name
        assignments.append(assigned)
    result = pd.concat(assignments, ignore_index=True).sort_values("image_id", ignore_index=True)
    assert len(result) == len(frame)
    assert result["image_id"].is_unique
    return result


def create_patch_manifest(split_frame: pd.DataFrame, patch_size: int = 256) -> pd.DataFrame:
    """Create patch coordinates after splitting, keeping every patch in its source split."""

    if patch_size <= 0:
        raise ValueError("Patch size must be positive")
    rows: list[dict[str, object]] = []
    for source in split_frame.itertuples(index=False):
        with Image.open(source.image_path) as image:
            width, height = image.size
        mask_paths = [
            Path(item) for item in str(source.mask_paths).split("|") if item and item != "nan"
        ]
        union_mask = np.zeros((height, width), dtype=bool)
        for mask_path in mask_paths:
            with Image.open(mask_path) as mask_image:
                union_mask |= np.asarray(mask_image.convert("L")) > 0
        has_segmentation_target = not source.is_defective or bool(mask_paths)
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch_width = min(patch_size, width - x)
                patch_height = min(patch_size, height - y)
                mask_pixels = int(union_mask[y : y + patch_height, x : x + patch_width].sum())
                rows.append(
                    {
                        "patch_id": f"{source.image_id}_x{x:04d}_y{y:04d}",
                        "image_id": source.image_id,
                        "split": source.split,
                        "x": x,
                        "y": y,
                        "width": patch_width,
                        "height": patch_height,
                        "image_path": source.image_path,
                        "mask_paths": source.mask_paths,
                        "source_is_defective": source.is_defective,
                        "defect_name": source.defect_name,
                        "fabric_code": source.fabric_code,
                        "has_segmentation_target": has_segmentation_target,
                        "mask_pixels": mask_pixels,
                        "is_positive": mask_pixels > 0,
                    }
                )
    patches = pd.DataFrame(rows)
    if not patches.empty:
        source_splits = patches.groupby("image_id")["split"].nunique()
        if (source_splits != 1).any():
            raise AssertionError("Patch leakage detected across source-image splits")
    return patches


def save_manifests(
    splits: pd.DataFrame, output_dir: Path, patch_size: int = 256
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "splits.csv"
    patch_path = output_dir / "patches.csv"
    splits.to_csv(split_path, index=False)
    create_patch_manifest(splits, patch_size=patch_size).to_csv(patch_path, index=False)
    return split_path, patch_path
