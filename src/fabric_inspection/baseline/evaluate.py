"""Validation-tuned evaluation for the classical texture baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fabric_inspection.baseline.local_texture import anomaly_score, postprocess_mask
from fabric_inspection.data.aitex import AitexRecord, load_grayscale, load_union_mask
from fabric_inspection.evaluation.metrics import (
    classification_metrics,
    segmentation_counts,
    segmentation_metrics_from_counts,
)


@dataclass(frozen=True)
class BaselineParameters:
    gaussian_sigma: float
    local_sigma: float
    morphology_kernel: int
    min_component_area: int


def _record_from_row(row: object) -> AitexRecord:
    mask_value = row.mask_paths
    mask_paths = tuple(Path(item) for item in str(mask_value).split("|") if item and item != "nan")
    return AitexRecord(
        image_id=row.image_id,
        image_path=Path(row.image_path),
        is_defective=bool(row.is_defective),
        defect_code=str(row.defect_code).zfill(3),
        defect_name=row.defect_name,
        fabric_code=str(row.fabric_code).zfill(2),
        mask_paths=mask_paths,
    )


def _load_partition(
    manifest: pd.DataFrame, split: str, parameters: BaselineParameters
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    subset = manifest[manifest["split"] == split]
    for row in subset.itertuples(index=False):
        record = _record_from_row(row)
        image = load_grayscale(record.image_path)
        target = load_union_mask(record, image.shape)
        examples.append(
            {
                "record": record,
                "image": image,
                "target": target,
                "has_segmentation_target": not record.is_defective or bool(record.mask_paths),
                "score": anomaly_score(
                    image,
                    gaussian_sigma=parameters.gaussian_sigma,
                    local_sigma=parameters.local_sigma,
                ),
            }
        )
    return examples


def _evaluate_threshold(
    examples: list[dict[str, object]], threshold: float, parameters: BaselineParameters
) -> tuple[dict[str, float | int], list[np.ndarray]]:
    totals = np.zeros(4, dtype=np.int64)
    predictions: list[np.ndarray] = []
    for example in examples:
        prediction = postprocess_mask(
            example["score"],
            threshold,
            parameters.morphology_kernel,
            parameters.min_component_area,
        )
        predictions.append(prediction)
        if example["has_segmentation_target"]:
            totals += segmentation_counts(prediction, example["target"])
    metrics = segmentation_metrics_from_counts(*totals.tolist())
    return metrics, predictions


def _select_pixel_threshold(
    examples: list[dict[str, object]], candidates: list[float], parameters: BaselineParameters
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in candidates:
        metrics, _ = _evaluate_threshold(examples, threshold, parameters)
        rows.append({"threshold": threshold, **metrics})
    search = pd.DataFrame(rows).sort_values("threshold", ignore_index=True)
    best = search.sort_values(
        ["dice", "pixel_recall", "threshold"], ascending=[False, False, True]
    ).iloc[0]
    return float(best["threshold"]), search


def _select_image_threshold(
    examples: list[dict[str, object]], predictions: list[np.ndarray]
) -> tuple[float, pd.DataFrame]:
    targets = [bool(example["record"].is_defective) for example in examples]
    areas = [float(prediction.mean()) for prediction in predictions]
    # Exhaustively consider validation-derived operating points. The test set has
    # no influence on either the candidates or the selected threshold.
    candidates = sorted({0.0, *areas, np.nextafter(max(areas), np.inf)})
    rows = []
    for threshold in candidates:
        metrics = classification_metrics(targets, [area >= threshold for area in areas], areas)
        rows.append({"image_area_threshold": threshold, **metrics})
    search = pd.DataFrame(rows)
    best = search.sort_values(
        ["f1", "recall", "image_area_threshold"], ascending=[False, False, True]
    ).iloc[0]
    return float(best["image_area_threshold"]), search


def _evaluate_partition(
    examples: list[dict[str, object]],
    pixel_threshold: float,
    image_threshold: float,
    parameters: BaselineParameters,
) -> tuple[dict[str, object], pd.DataFrame, list[np.ndarray]]:
    segmentation, predictions = _evaluate_threshold(examples, pixel_threshold, parameters)
    rows = []
    for example, prediction in zip(examples, predictions, strict=True):
        record = example["record"]
        area = float(prediction.mean())
        if example["has_segmentation_target"]:
            counts = segmentation_counts(prediction, example["target"])
            per_image = segmentation_metrics_from_counts(*counts)
        else:
            per_image = {"dice": None, "iou": None, "pixel_recall": None}
        rows.append(
            {
                "image_id": record.image_id,
                "defect_name": record.defect_name,
                "target_defective": record.is_defective,
                "predicted_defective": area >= image_threshold,
                "predicted_area_fraction": area,
                "dice": per_image["dice"],
                "iou": per_image["iou"],
                "pixel_recall": per_image["pixel_recall"],
            }
        )
    frame = pd.DataFrame(rows)
    classification = classification_metrics(
        frame["target_defective"],
        frame["predicted_defective"],
        frame["predicted_area_fraction"],
    )
    return {"segmentation": segmentation, "classification": classification}, frame, predictions


def _save_examples(
    examples: list[dict[str, object]],
    predictions: list[np.ndarray],
    rows: pd.DataFrame,
    output_path: Path,
) -> None:
    candidates = rows[rows["target_defective"] & rows["dice"].notna()].copy()
    if candidates.empty:
        return
    selected_indices = list(candidates.nlargest(2, "dice").index) + list(
        candidates.nsmallest(2, "dice").index
    )
    figure, axes = plt.subplots(len(selected_indices), 4, figsize=(16, 3 * len(selected_indices)))
    axes = np.atleast_2d(axes)
    for plot_row, index in enumerate(selected_indices):
        example = examples[index]
        prediction = predictions[index]
        image = example["image"]
        target = example["target"]
        record = example["record"]
        axes[plot_row, 0].imshow(image, cmap="gray")
        axes[plot_row, 0].set_title(record.image_id)
        axes[plot_row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[plot_row, 1].set_title("Ground truth")
        axes[plot_row, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[plot_row, 2].set_title(f"Prediction · Dice {rows.loc[index, 'dice']:.3f}")
        axes[plot_row, 3].imshow(image, cmap="gray")
        axes[plot_row, 3].imshow(prediction, cmap="Reds", alpha=np.where(prediction, 0.55, 0.0))
        axes[plot_row, 3].set_title("Prediction overlay")
        for axis in axes[plot_row]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_baseline(config: dict[str, object]) -> dict[str, object]:
    data_config = config["data"]
    baseline_config = config["baseline"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = BaselineParameters(
        gaussian_sigma=float(baseline_config["gaussian_sigma"]),
        local_sigma=float(baseline_config["local_sigma"]),
        morphology_kernel=int(baseline_config["morphology_kernel"]),
        min_component_area=int(baseline_config["min_component_area"]),
    )
    manifest = pd.read_csv(data_config["split_manifest"])
    validation = _load_partition(manifest, "validation", parameters)
    pixel_threshold, pixel_search = _select_pixel_threshold(
        validation, list(baseline_config["threshold_candidates"]), parameters
    )
    validation_segmentation, validation_predictions = _evaluate_threshold(
        validation, pixel_threshold, parameters
    )
    image_threshold, image_search = _select_image_threshold(
        validation,
        validation_predictions,
    )
    validation_metrics, validation_rows, _ = _evaluate_partition(
        validation, pixel_threshold, image_threshold, parameters
    )
    test = _load_partition(manifest, "test", parameters)
    test_metrics, test_rows, test_predictions = _evaluate_partition(
        test, pixel_threshold, image_threshold, parameters
    )
    result = {
        "selection_split": "validation",
        "pixel_threshold": pixel_threshold,
        "image_area_threshold": image_threshold,
        "parameters": parameters.__dict__,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    pixel_search.to_csv(output_dir / "pixel_threshold_search.csv", index=False)
    image_search.to_csv(output_dir / "image_threshold_search.csv", index=False)
    validation_rows.to_csv(output_dir / "validation_predictions.csv", index=False)
    test_rows.to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _save_examples(test, test_predictions, test_rows, output_dir / "test_examples.png")
    return result
