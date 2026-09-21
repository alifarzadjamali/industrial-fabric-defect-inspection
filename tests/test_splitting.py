from pathlib import Path

import pandas as pd
import pytest

from fabric_inspection.data.aitex import AitexRecord
from fabric_inspection.data.splitting import create_patch_manifest, create_split_manifest


def _records() -> list[AitexRecord]:
    records = []
    for fabric in range(1, 8):
        for index in range(20):
            defective = index >= 10
            code = "002" if defective else "000"
            image_id = f"{fabric:02d}{index:02d}_{code}_{fabric:02d}"
            records.append(
                AitexRecord(
                    image_id=image_id,
                    image_path=Path(f"/{image_id}.png"),
                    is_defective=defective,
                    defect_code=code,
                    defect_name="broken_end" if defective else "normal",
                    fabric_code=f"{fabric:02d}",
                    mask_paths=(),
                )
            )
    return records


def test_source_images_are_unique_and_split_is_reproducible() -> None:
    first = create_split_manifest(_records(), seed=42)
    second = create_split_manifest(_records(), seed=42)
    assert first.equals(second)
    assert first["image_id"].is_unique
    assert set(first["split"]) == {"train", "validation", "test"}


def test_patch_manifest_rejects_non_positive_patch_size() -> None:
    with pytest.raises(ValueError, match="Patch size must be positive"):
        create_patch_manifest(pd.DataFrame(), patch_size=0)


def test_split_manifest_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="At least one source image"):
        create_split_manifest([])
