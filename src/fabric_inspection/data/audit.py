"""Dataset integrity checks, statistics, and visual audit outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from fabric_inspection.data.aitex import AitexRecord, load_grayscale, load_union_mask, sha256_file


def _connected_components(mask: np.ndarray) -> tuple[int, list[int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(int).tolist() if count > 1 else []
    return count - 1, areas


def _save_examples(rows: pd.DataFrame, records_by_id: dict[str, AitexRecord], output: Path) -> None:
    defective = rows[(rows["is_defective"]) & (rows["defective_pixels"] > 0)]
    if defective.empty:
        return
    selected = pd.concat(
        [defective.nsmallest(2, "defective_pixels"), defective.nlargest(2, "defective_pixels")]
    )
    figure, axes = plt.subplots(len(selected), 3, figsize=(15, 3.2 * len(selected)), squeeze=False)
    for row_index, (_, row) in enumerate(selected.iterrows()):
        record = records_by_id[row["image_id"]]
        image = load_grayscale(record.image_path)
        mask = load_union_mask(record, image.shape)
        axes[row_index, 0].imshow(image, cmap="gray")
        axes[row_index, 0].set_title(f"{record.image_id}: {record.defect_name}")
        axes[row_index, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title(f"Mask ({int(mask.sum()):,} px)")
        axes[row_index, 2].imshow(image, cmap="gray")
        axes[row_index, 2].imshow(mask, cmap="Reds", alpha=np.where(mask, 0.55, 0.0))
        axes[row_index, 2].set_title("Ground-truth overlay")
        for axis in axes[row_index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def audit_dataset(records: list[AitexRecord], output_dir: Path) -> dict[str, object]:
    """Audit image/mask integrity and save machine-readable and visual reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    corrupt_files: list[str] = []
    dimension_mismatches: list[str] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    perceptual: dict[str, str] = {}

    for record in records:
        try:
            image = load_grayscale(record.image_path)
            with Image.open(record.image_path) as source:
                perceptual[record.image_id] = str(imagehash.phash(source.convert("L")))
        except (OSError, UnidentifiedImageError) as error:
            corrupt_files.append(f"{record.image_path}: {error}")
            continue
        hashes[sha256_file(record.image_path)].append(record.image_id)
        defective_pixels = 0
        component_areas: list[int] = []
        try:
            mask = load_union_mask(record, image.shape)
            defective_pixels = int(mask.sum())
            _, component_areas = _connected_components(mask)
        except ValueError as error:
            dimension_mismatches.append(str(error))
        rows.append(
            {
                "image_id": record.image_id,
                "is_defective": record.is_defective,
                "defect_code": record.defect_code,
                "defect_name": record.defect_name,
                "fabric_code": record.fabric_code,
                "width": image.shape[1],
                "height": image.shape[0],
                "mask_count": len(record.mask_paths),
                "defective_pixels": defective_pixels,
                "defective_pixel_fraction": defective_pixels / image.size,
                "defect_components": len(component_areas),
                "smallest_component_pixels": min(component_areas, default=0),
                "largest_component_pixels": max(component_areas, default=0),
                "sha256": sha256_file(record.image_path),
                "phash": perceptual[record.image_id],
            }
        )

    frame = pd.DataFrame(rows)
    exact_duplicates = [ids for ids in hashes.values() if len(ids) > 1]
    near_duplicates: list[dict[str, object]] = []
    ids = sorted(perceptual)
    parsed_hashes = {key: imagehash.hex_to_hash(value) for key, value in perceptual.items()}
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            distance = parsed_hashes[left] - parsed_hashes[right]
            if distance <= 2:
                near_duplicates.append({"left": left, "right": right, "phash_distance": distance})

    missing_masks = [
        record.image_id for record in records if record.is_defective and not record.mask_paths
    ]
    unexpected_masks = [
        record.image_id for record in records if not record.is_defective and record.mask_paths
    ]
    summary: dict[str, object] = {
        "total_images": len(records),
        "readable_images": len(frame),
        "defective_images": int(frame["is_defective"].sum()),
        "normal_images": int((~frame["is_defective"]).sum()),
        "total_masks": int(frame["mask_count"].sum()),
        "image_dimensions": frame.groupby(["width", "height"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records"),
        "defect_category_distribution": frame[frame["is_defective"]]["defect_name"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "fabric_distribution": frame["fabric_code"].value_counts().sort_index().to_dict(),
        "overall_defective_pixel_fraction": float(
            frame["defective_pixels"].sum() / (frame["width"] * frame["height"]).sum()
        ),
        "defective_image_pixel_fraction": {
            "min": float(frame.loc[frame["is_defective"], "defective_pixel_fraction"].min()),
            "median": float(frame.loc[frame["is_defective"], "defective_pixel_fraction"].median()),
            "max": float(frame.loc[frame["is_defective"], "defective_pixel_fraction"].max()),
        },
        "missing_masks": missing_masks,
        "unexpected_masks": unexpected_masks,
        "corrupt_files": corrupt_files,
        "dimension_mismatches": dimension_mismatches,
        "exact_duplicate_groups": exact_duplicates,
        "near_duplicate_pairs_phash_distance_le_2": near_duplicates,
    }
    frame.to_csv(output_dir / "image_inventory.csv", index=False)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _save_examples(
        frame, {record.image_id: record for record in records}, output_dir / "mask_examples.png"
    )
    return summary
