"""Held-out error analysis and controlled camera-variation robustness tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from fabric_inspection.data.aitex import AitexRecord, load_grayscale, load_union_mask
from fabric_inspection.evaluation.metrics import (
    segmentation_counts,
    segmentation_metrics_from_counts,
)
from fabric_inspection.evaluation.model_evaluator import (
    ImagePrediction,
    evaluate_images,
    load_evaluation_config,
    load_model,
    predict_original_images,
)
from fabric_inspection.inference.tiling import predict_grayscale_image
from fabric_inspection.training.trainer import seed_everything


@dataclass(frozen=True)
class AnalysisRuntime:
    device: torch.device
    model: torch.nn.Module
    segmentation_threshold: float
    image_threshold: float
    image_size: int
    batch_size: int
    mixed_precision: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_from_row(row: object) -> AitexRecord:
    masks = tuple(Path(item) for item in str(row.mask_paths).split("|") if item and item != "nan")
    defective = bool(row.is_defective)
    return AitexRecord(
        image_id=str(row.image_id),
        image_path=Path(row.image_path),
        is_defective=defective,
        defect_code=str(row.defect_code).zfill(3),
        defect_name=str(row.defect_name),
        fabric_code=str(row.fabric_code).zfill(2),
        mask_paths=masks,
    )


def _load_records(split_manifest: Path, split: str = "test") -> list[AitexRecord]:
    frame = pd.read_csv(split_manifest)
    return [_record_from_row(row) for row in frame[frame["split"] == split].itertuples(index=False)]


def _load_runtime(evaluation_config_path: Path, seed: int) -> tuple[AnalysisRuntime, dict]:
    evaluation = load_evaluation_config(evaluation_config_path)
    output_dir = Path(evaluation["output_dir"])
    frozen = json.loads((output_dir / "frozen_thresholds.json").read_text(encoding="utf-8"))
    checkpoint_path = Path(evaluation["checkpoint"])
    if _sha256(checkpoint_path) != frozen["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from the validation-frozen evaluation checkpoint")
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(checkpoint_path, device)
    runtime = AnalysisRuntime(
        device=device,
        model=model,
        segmentation_threshold=float(frozen["segmentation_threshold"]),
        image_threshold=float(frozen["image_component_threshold"]),
        image_size=int(evaluation["image_size"]),
        batch_size=int(evaluation["batch_size"]),
        mixed_precision=bool(evaluation["mixed_precision"]),
    )
    return runtime, evaluation


def _defect_features(record: AitexRecord, target: np.ndarray) -> dict[str, float | str]:
    image = load_grayscale(record.image_path)
    pixels = int(target.sum())
    if pixels:
        y_coordinates, x_coordinates = np.nonzero(target)
        dilated = cv2.dilate(target.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
        ring = dilated & ~target
        local_contrast = (
            float(abs(image[target].mean() - image[ring].mean())) if ring.any() else 0.0
        )
        centroid_x = float(x_coordinates.mean() / image.shape[1])
        centroid_y = float(y_coordinates.mean() / image.shape[0])
    else:
        local_contrast = centroid_x = centroid_y = float("nan")
    if pixels <= 32:
        size_bin = "tiny_1_32"
    elif pixels <= 256:
        size_bin = "small_33_256"
    elif pixels <= 4096:
        size_bin = "medium_257_4096"
    else:
        size_bin = "large_over_4096"
    return {
        "fabric_code": record.fabric_code,
        "ground_truth_pixels": pixels,
        "ground_truth_fraction": pixels / target.size,
        "size_bin": size_bin,
        "local_contrast": local_contrast,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "image_standard_deviation": float(image.std()),
    }


def _correlation(frame: pd.DataFrame, feature: str) -> dict[str, float | None]:
    usable = frame[[feature, "dice"]].dropna()
    if len(usable) < 3 or usable[feature].nunique() < 2:
        return {"rho": None, "p_value": None}
    result = spearmanr(usable[feature], usable["dice"])
    return {"rho": float(result.statistic), "p_value": float(result.pvalue)}


def _save_case_grid(
    case_ids: list[str],
    predictions: dict[str, ImagePrediction],
    records: dict[str, AitexRecord],
    threshold: float,
    title: str,
    path: Path,
) -> None:
    if not case_ids:
        return
    figure, axes = plt.subplots(len(case_ids), 4, figsize=(17, 3.1 * len(case_ids)), squeeze=False)
    figure.suptitle(title)
    for row_index, image_id in enumerate(case_ids):
        prediction = predictions[image_id]
        image = load_grayscale(records[image_id].image_path)
        binary = prediction.probability >= threshold
        dice_row = segmentation_metrics_from_counts(*segmentation_counts(binary, prediction.target))
        axes[row_index, 0].imshow(image, cmap="gray")
        axes[row_index, 0].set_title(image_id)
        axes[row_index, 1].imshow(prediction.target, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title("Ground truth")
        axes[row_index, 2].imshow(binary, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].set_title(f"Prediction · Dice {dice_row['dice']:.3f}")
        axes[row_index, 3].imshow(image, cmap="gray")
        axes[row_index, 3].imshow(binary, cmap="Reds", alpha=np.where(binary, 0.55, 0.0))
        axes[row_index, 3].set_title("Prediction overlay")
        for axis in axes[row_index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_error_analysis(
    analysis_config: dict[str, object], runtime: AnalysisRuntime, evaluation: dict[str, object]
) -> tuple[list[ImagePrediction], dict[str, object]]:
    output_dir = Path(analysis_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(Path(analysis_config["split_manifest"]))
    predictions = predict_original_images(
        runtime.model,
        Path(evaluation["patch_manifest"]),
        "test",
        runtime.image_size,
        runtime.batch_size,
        int(evaluation["num_workers"]),
        runtime.device,
        runtime.mixed_precision,
    )
    metrics, per_image, _ = evaluate_images(
        predictions, runtime.segmentation_threshold, runtime.image_threshold
    )
    record_map = {record.image_id: record for record in records}
    prediction_map = {prediction.image_id: prediction for prediction in predictions}
    features = pd.DataFrame(
        [
            {"image_id": item.image_id, **_defect_features(record_map[item.image_id], item.target)}
            for item in predictions
        ]
    )
    detailed = per_image.merge(features, on="image_id", validate="one_to_one")
    detailed["classification_correct"] = (
        detailed["target_defective"] == detailed["predicted_defective"]
    )
    detailed.to_csv(output_dir / "error_inventory.csv", index=False)
    defective = detailed[detailed["target_defective"]].copy()
    grouped_size = (
        defective.groupby("size_bin", observed=True)
        .agg(
            images=("image_id", "count"),
            mean_dice=("dice", "mean"),
            detection_recall=("predicted_defective", "mean"),
        )
        .reset_index()
    )
    grouped_size.to_csv(output_dir / "performance_by_size.csv", index=False)
    grouped_fabric = (
        detailed.groupby("fabric_code", observed=True)
        .agg(
            images=("image_id", "count"),
            mean_dice=("dice", "mean"),
            image_accuracy=("classification_correct", "mean"),
        )
        .reset_index()
    )
    grouped_fabric.to_csv(output_dir / "performance_by_fabric.csv", index=False)

    correlations = {
        "log_ground_truth_pixels_vs_dice": _correlation(
            defective.assign(log_ground_truth_pixels=np.log1p(defective["ground_truth_pixels"])),
            "log_ground_truth_pixels",
        ),
        "local_contrast_vs_dice": _correlation(defective, "local_contrast"),
        "horizontal_location_vs_dice": _correlation(defective, "centroid_x"),
        "vertical_location_vs_dice": _correlation(defective, "centroid_y"),
    }
    tp = detailed[detailed["target_defective"] & detailed["predicted_defective"]]
    fp = detailed[~detailed["target_defective"] & detailed["predicted_defective"]]
    fn = detailed[detailed["target_defective"] & ~detailed["predicted_defective"]]
    cases = {
        "true_positives": tp.sort_values("dice", ascending=False)["image_id"].head(4).tolist(),
        "false_positives": fp.sort_values("largest_component_fraction", ascending=False)["image_id"]
        .head(4)
        .tolist(),
        "false_negatives": fn.sort_values("ground_truth_pixels")["image_id"].head(4).tolist(),
        "best_segmentations": defective.sort_values("dice", ascending=False)["image_id"]
        .head(4)
        .tolist(),
        "worst_segmentations": defective.sort_values("dice")["image_id"].head(4).tolist(),
    }
    for name, image_ids in cases.items():
        _save_case_grid(
            image_ids,
            prediction_map,
            record_map,
            runtime.segmentation_threshold,
            name.replace("_", " ").title(),
            output_dir / f"{name}.png",
        )
    summary = {
        "metrics": metrics,
        "case_counts": {
            "true_positives": len(tp),
            "false_positives": len(fp),
            "false_negatives": len(fn),
            "true_negatives": int(
                (~detailed["target_defective"] & ~detailed["predicted_defective"]).sum()
            ),
        },
        "selected_cases": cases,
        "correlations": correlations,
        "performance_by_size": grouped_size.to_dict("records"),
    }
    (output_dir / "error_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return predictions, summary


def _transform(
    name: str,
    image: np.ndarray,
    target: np.ndarray,
    robustness: dict[str, object],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    values = image.astype(np.float32)
    if name.startswith("brightness_"):
        factor = float(name.split("_")[1])
        return np.clip(values * factor, 0, 255).astype(np.uint8), target
    if name.startswith("contrast_"):
        factor = float(name.split("_")[1])
        adjusted = (values - values.mean()) * factor + values.mean()
        return np.clip(adjusted, 0, 255).astype(np.uint8), target
    if name == "blur":
        sigma = float(robustness["blur_sigma"])
        return cv2.GaussianBlur(image, (0, 0), sigma), target
    if name == "noise":
        noise = rng.normal(0.0, float(robustness["noise_standard_deviation"]), image.shape)
        return np.clip(values + noise, 0, 255).astype(np.uint8), target
    if name == "resize":
        factor = float(robustness["resize_factor"])
        height, width = image.shape
        smaller = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        return cv2.resize(smaller, (width, height), interpolation=cv2.INTER_LINEAR), target
    if name.startswith("rotation_"):
        angle = float(name.split("_")[1])
        height, width = image.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated_image = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        rotated_target = cv2.warpAffine(
            target.astype(np.uint8),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(bool)
        return rotated_image, rotated_target
    raise ValueError(f"Unknown robustness condition: {name}")


def _condition_names(robustness: dict[str, object]) -> list[str]:
    return [
        "original",
        *(f"brightness_{value}" for value in robustness["brightness_factors"]),
        *(f"contrast_{value}" for value in robustness["contrast_factors"]),
        "blur",
        "noise",
        "resize",
        *(f"rotation_{value}" for value in robustness["rotation_degrees"]),
    ]


def _plot_robustness(frame: pd.DataFrame, path: Path) -> None:
    positions = np.arange(len(frame))
    width = 0.27
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(positions - width, frame["dice"], width, label="Dice")
    axis.bar(positions, frame["pixel_recall"], width, label="Pixel recall")
    axis.bar(positions + width, frame["image_f1"], width, label="Image F1")
    axis.set(
        xticks=positions,
        xticklabels=frame["condition"],
        ylabel="Metric",
        ylim=(0, 1),
    )
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_robustness_analysis(
    analysis_config: dict[str, object],
    runtime: AnalysisRuntime,
    original_predictions: list[ImagePrediction],
) -> pd.DataFrame:
    output_dir = Path(analysis_config["output_dir"])
    records = _load_records(Path(analysis_config["split_manifest"]))
    original_map = {item.image_id: item for item in original_predictions}
    robustness = analysis_config["robustness"]
    rows: list[dict[str, object]] = []
    for condition in _condition_names(robustness):
        transformed_predictions: list[ImagePrediction] = []
        for record in records:
            if condition == "original":
                transformed_predictions.append(original_map[record.image_id])
                continue
            image = load_grayscale(record.image_path)
            target = load_union_mask(record, image.shape)
            stable_seed = int.from_bytes(
                hashlib.sha256(f"{condition}:{record.image_id}".encode()).digest()[:8], "little"
            )
            transformed_image, transformed_target = _transform(
                condition,
                image,
                target,
                robustness,
                np.random.default_rng(stable_seed),
            )
            probability = predict_grayscale_image(
                runtime.model,
                transformed_image,
                runtime.device,
                runtime.image_size,
                runtime.batch_size,
                runtime.mixed_precision,
            )
            transformed_predictions.append(
                ImagePrediction(
                    record.image_id,
                    record.defect_name,
                    record.is_defective,
                    probability,
                    transformed_target,
                )
            )
        metrics, _, _ = evaluate_images(
            transformed_predictions, runtime.segmentation_threshold, runtime.image_threshold
        )
        rows.append(
            {
                "condition": condition,
                "dice": metrics["segmentation"]["dice"],
                "iou": metrics["segmentation"]["iou"],
                "pixel_precision": metrics["segmentation"]["pixel_precision"],
                "pixel_recall": metrics["segmentation"]["pixel_recall"],
                "macro_defective_dice": metrics["segmentation"]["macro_defective_dice"],
                "image_accuracy": metrics["classification"]["accuracy"],
                "image_precision": metrics["classification"]["precision"],
                "image_recall": metrics["classification"]["recall"],
                "image_f1": metrics["classification"]["f1"],
                "image_roc_auc": metrics["classification"]["roc_auc"],
            }
        )
    frame = pd.DataFrame(rows)
    original = frame.iloc[0]
    frame["dice_delta"] = frame["dice"] - original["dice"]
    frame["image_f1_delta"] = frame["image_f1"] - original["image_f1"]
    frame.to_csv(output_dir / "robustness_metrics.csv", index=False)
    (output_dir / "robustness_metrics.json").write_text(
        frame.to_json(orient="records", indent=2), encoding="utf-8"
    )
    _plot_robustness(frame, output_dir / "robustness_summary.png")
    return frame


def run_analysis(config_path: Path) -> dict[str, object]:
    with config_path.open(encoding="utf-8") as handle:
        analysis_config = yaml.safe_load(handle)
    runtime, evaluation = _load_runtime(
        Path(analysis_config["evaluation_config"]), int(analysis_config["seed"])
    )
    predictions, error_summary = run_error_analysis(analysis_config, runtime, evaluation)
    robustness = run_robustness_analysis(analysis_config, runtime, predictions)
    return {
        "error_summary": error_summary,
        "robustness": robustness.to_dict("records"),
    }
