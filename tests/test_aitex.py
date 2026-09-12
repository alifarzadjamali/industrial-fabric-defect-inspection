from pathlib import Path

import numpy as np
from PIL import Image

from fabric_inspection.data.aitex import discover_records, load_union_mask


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
