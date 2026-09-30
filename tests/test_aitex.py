from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fabric_inspection.data.aitex import discover_records, load_union_mask, sha256_file
from fabric_inspection.data.dataset import AitexPatchDataset


def test_sha256_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError, match="Chunk size must be positive"):
        sha256_file(Path("unused"), chunk_size=0)


def test_discovery_associates_and_unions_multiple_masks(tmp_path: Path) -> None:
    defect_dir = tmp_path / "Defect_images"
    mask_dir = tmp_path / "Mask_images"
    defect_dir.mkdir()
    mask_dir.mkdir()
    image = np.zeros((8, 16), dtype=np.uint8)
    first = image.copy()
    second = image.copy()
    first[1:3, 1:3] = 255
    second[5:7, 10:12] = 255
    Image.fromarray(image).save(defect_dir / "0001_002_01.png")
    Image.fromarray(first).save(mask_dir / "0001_002_01_mask.png")
    Image.fromarray(second).save(mask_dir / "0001_002_01_mask1.png")

    records = discover_records(tmp_path)
    assert len(records) == 1
    assert records[0].defect_name == "broken_end"
    assert len(records[0].mask_paths) == 2
    assert int(load_union_mask(records[0]).sum()) == 8


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("image_size", 0, "Image size must be positive"),
        ("source_size", 0, "Source size must be positive"),
        ("augmentation_profile", "unknown", "Unknown augmentation profile"),
    ],
)
def test_patch_dataset_validates_configuration_before_loading(
    argument: str, value: object, message: str
) -> None:
    manifest = pd.DataFrame(columns=["split", "has_segmentation_target"])
    with pytest.raises(ValueError, match=message):
        AitexPatchDataset(manifest, "train", **{argument: value})


def test_patch_dataset_materialises_patch_metadata_once(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.full((4, 4), 127, dtype=np.uint8)).save(image_path)
    Image.fromarray(np.eye(4, dtype=np.uint8) * 255).save(mask_path)
    manifest = pd.DataFrame(
        [
            {
                "split": "train",
                "has_segmentation_target": True,
                "is_positive": True,
                "x": 0,
                "y": 0,
                "width": 4,
                "height": 4,
                "image_path": image_path,
                "mask_paths": f"{mask_path}|",
            }
        ]
    )
    dataset = AitexPatchDataset(manifest, "train", image_size=4)
    dataset.rows.loc[0, "mask_paths"] = "missing.png"
    image, mask = dataset[0]
    assert image.shape == (3, 4, 4)
    assert int(mask.sum()) == 4


def test_patch_dataset_pads_single_pixel_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "narrow.png"
    Image.fromarray(np.array([[64, 128]], dtype=np.uint8)).save(image_path)
    manifest = pd.DataFrame(
        [
            {
                "split": "validation",
                "has_segmentation_target": True,
                "is_positive": False,
                "x": 0,
                "y": 0,
                "width": 2,
                "height": 1,
                "image_path": image_path,
                "mask_paths": "",
            }
        ]
    )
    dataset = AitexPatchDataset(manifest, "validation", image_size=4)
    image, mask = dataset[0]
    assert image.shape == (3, 4, 4)
    assert mask.shape == (1, 4, 4)
