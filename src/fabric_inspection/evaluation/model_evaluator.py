"""Leakage-safe full-image model evaluation and threshold selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from fabric_inspection.data.aitex import sha256_file
from fabric_inspection.data.dataset import AitexPatchDataset, _seed_worker
from fabric_inspection.evaluation.metrics import (
    classification_metrics,
    classification_threshold_metrics,
    segmentation_counts,
    segmentation_metrics_from_counts,
)
from fabric_inspection.inference.tiling import predict_grayscale_image
from fabric_inspection.models.unet import ResNet18UNet
from fabric_inspection.training.trainer import seed_everything


@dataclass
class ImagePrediction:
    image_id: str
    defect_name: str
    target_defective: bool
    probability: np.ndarray
    target: np.ndarray


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ResNet18UNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ResNet18UNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_original_images(
    model: ResNet18UNet,
    manifest_path: Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    mixed_precision: bool,
) -> list[ImagePrediction]:
    """Infer patches and reconstruct source-sized maps without cross-split access."""

    manifest = pd.read_csv(manifest_path)
    dataset = AitexPatchDataset(manifest, split, image_size=image_size, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        persistent_workers=num_workers > 0,
    )
    assembled: dict[str, ImagePrediction] = {}
    for image_id, rows in dataset.rows.groupby("image_id", sort=False):
        width = int((rows["x"] + rows["width"]).max())
        height = int((rows["y"] + rows["height"]).max())
        first = rows.iloc[0]
        assembled[image_id] = ImagePrediction(
            image_id=image_id,
            defect_name=str(first["defect_name"]),
            target_defective=bool(first["source_is_defective"]),
            probability=np.zeros((height, width), dtype=np.float32),
            target=np.zeros((height, width), dtype=bool),
        )

    offset = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                probabilities = torch.sigmoid(model(images))
            probabilities_np = probabilities.float().cpu().numpy()[:, 0]
            targets_np = targets.numpy()[:, 0] >= 0.5
            batch_rows = dataset.rows.iloc[offset : offset + len(images)]
            for local_index, row in enumerate(batch_rows.itertuples(index=False)):
                x, y = int(row.x), int(row.y)
                width, height = int(row.width), int(row.height)
                destination = assembled[row.image_id]
                destination.probability[y : y + height, x : x + width] = probabilities_np[
                    local_index, :height, :width
                ]
                destination.target[y : y + height, x : x + width] = targets_np[
                    local_index, :height, :width
                ]
            offset += len(images)
    if offset != len(dataset):
        raise AssertionError(f"Reconstructed {offset} of {len(dataset)} patches")
    return list(assembled.values())


def predict_source_images(
    model: ResNet18UNet,
    manifest_path: Path,
    split: str,
    image_size: int,
    tile_size: int,
    overlap: int,
    batch_size: int,
    device: torch.device,
    mixed_precision: bool,
) -> list[ImagePrediction]:
    """Infer directly from a source manifest, including overlap and scale changes."""

    manifest = pd.read_csv(manifest_path)
    usable = (
        manifest["has_segmentation_target"].astype(bool)
        if "has_segmentation_target" in manifest
        else (~manifest["is_defective"].astype(bool) | manifest["mask_paths"].notna())
    )
    selected = manifest[(manifest["split"] == split) & usable]
    predictions: list[ImagePrediction] = []
    for row in selected.itertuples(index=False):
        with Image.open(row.image_path) as image_file:
            image = np.asarray(image_file.convert("L"))
        target = np.zeros_like(image, dtype=bool)
        for value in str(row.mask_paths).split("|"):
            if value and value != "nan":
                with Image.open(value) as mask_file:
                    target |= np.asarray(mask_file.convert("L")) > 0
        probability = predict_grayscale_image(
            model,
            image,
            device,
            image_size=image_size,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            mixed_precision=mixed_precision,
        )
        predictions.append(
            ImagePrediction(
                image_id=str(row.image_id),
                defect_name=str(row.defect_name),
                target_defective=bool(row.is_defective),
                probability=probability,
                target=target,
            )
        )
    return predictions


def _predict_configured(
    config: dict[str, object], model: ResNet18UNet, split: str, device: torch.device
) -> list[ImagePrediction]:
    if "split_manifest" in config:
        return predict_source_images(
            model,
            Path(config["split_manifest"]),
            split,
            int(config["image_size"]),
            int(config.get("tile_size", config["image_size"])),
            int(config.get("overlap", 0)),
            int(config["batch_size"]),
            device,
            bool(config["mixed_precision"]),
        )
    return predict_original_images(
        model,
        Path(config["patch_manifest"]),
        split,
        int(config["image_size"]),
        int(config["batch_size"]),
        int(config["num_workers"]),
        device,
        bool(config["mixed_precision"]),
    )


def _segmentation_summary(
    images: list[ImagePrediction], threshold: float
) -> dict[str, float | int]:
    totals = np.zeros(4, dtype=np.int64)
    defective_dice: list[float] = []
    for image in images:
        prediction = image.probability >= threshold
        counts = segmentation_counts(prediction, image.target)
        totals += counts
        if image.target_defective:
            defective_dice.append(float(segmentation_metrics_from_counts(*counts)["dice"]))
    metrics = segmentation_metrics_from_counts(*totals.tolist())
    metrics["macro_defective_dice"] = float(np.mean(defective_dice)) if defective_dice else 0.0
    return metrics


def select_segmentation_threshold(
    images: list[ImagePrediction], candidates: list[float]
) -> tuple[float, pd.DataFrame]:
    if not candidates:
        raise ValueError("At least one segmentation threshold candidate is required")
    thresholds = np.asarray(candidates, dtype=np.float64)
    if not np.isfinite(thresholds).all():
        raise ValueError("Segmentation threshold candidates must be finite")

    totals = np.zeros((len(thresholds), 4), dtype=np.int64)
    defective_dice = np.zeros(len(thresholds), dtype=np.float64)
    defective_count = 0
    for image in images:
        probability = np.asarray(image.probability)
        target = np.asarray(image.target, dtype=bool)
        if probability.shape != target.shape:
            raise ValueError(f"Shape mismatch: {probability.shape} != {target.shape}")
        if not np.isfinite(probability).all():
            raise ValueError("Segmentation probabilities must be finite")

        order = np.argsort(probability, axis=None)
        sorted_probability = probability.ravel()[order]
        sorted_target = target.ravel()[order]
        positive_prefix = np.concatenate(([0], np.cumsum(sorted_target, dtype=np.int64)))
        # NumPy casts a scalar threshold to the probability array's dtype for
        # direct comparisons; mirror that behavior at representable boundaries.
        image_thresholds = thresholds.astype(probability.dtype, copy=False)
        split_indices = np.searchsorted(sorted_probability, image_thresholds, side="left")
        positive_total = int(positive_prefix[-1])
        true_positive = positive_total - positive_prefix[split_indices]
        predicted_positive = probability.size - split_indices
        false_positive = predicted_positive - true_positive
        false_negative = positive_total - true_positive
        true_negative = split_indices - positive_prefix[split_indices]
        counts = np.column_stack(
            (true_positive, false_positive, false_negative, true_negative)
        )
        totals += counts
        if image.target_defective:
            denominators = 2 * true_positive + false_positive + false_negative
            defective_dice += np.divide(
                2 * true_positive,
                denominators,
                out=np.ones(len(thresholds), dtype=np.float64),
                where=denominators != 0,
            )
            defective_count += 1

    rows = []
    for index, threshold in enumerate(candidates):
        metrics = segmentation_metrics_from_counts(*totals[index].tolist())
        metrics["macro_defective_dice"] = (
            float(defective_dice[index] / defective_count) if defective_count else 0.0
        )
        rows.append({"segmentation_threshold": threshold, **metrics})
    search = pd.DataFrame(rows).sort_values("segmentation_threshold", ignore_index=True)
    best = search.sort_values(
        ["dice", "pixel_recall", "segmentation_threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    return float(best["segmentation_threshold"]), search


def largest_component_fraction(mask: np.ndarray) -> float:
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if component_count > 1 else 0
    return largest / mask.size


def select_image_threshold(
    images: list[ImagePrediction], segmentation_threshold: float
) -> tuple[float, pd.DataFrame]:
    targets = [image.target_defective for image in images]
    scores = [
        largest_component_fraction(image.probability >= segmentation_threshold) for image in images
    ]
    search = pd.DataFrame(
        classification_threshold_metrics(targets, scores, "image_component_threshold")
    )
    best = search.sort_values(
        ["f1", "recall", "accuracy", "image_component_threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return float(best["image_component_threshold"]), search


def evaluate_images(
    images: list[ImagePrediction], segmentation_threshold: float, image_threshold: float
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    category_counts: dict[str, np.ndarray] = {}
    totals = np.zeros(4, dtype=np.int64)
    defective_dice: list[float] = []
    for image in images:
        prediction = image.probability >= segmentation_threshold
        counts = segmentation_counts(prediction, image.target)
        metrics = segmentation_metrics_from_counts(*counts)
        totals += counts
        if image.target_defective:
            defective_dice.append(float(metrics["dice"]))
        score = largest_component_fraction(prediction)
        rows.append(
            {
                "image_id": image.image_id,
                "defect_name": image.defect_name,
                "target_defective": image.target_defective,
                "predicted_defective": score >= image_threshold,
                "largest_component_fraction": score,
                **metrics,
            }
        )
        counts_for_category = category_counts.get(image.defect_name)
        if counts_for_category is None:
            counts_for_category = category_counts[image.defect_name] = np.zeros(4, dtype=np.int64)
        counts_for_category += counts
    per_image = pd.DataFrame(rows)
    segmentation = segmentation_metrics_from_counts(*totals.tolist())
    segmentation["macro_defective_dice"] = (
        float(np.mean(defective_dice)) if defective_dice else 0.0
    )
    classification = classification_metrics(
        per_image["target_defective"],
        per_image["predicted_defective"],
        per_image["largest_component_fraction"],
    )
    categories = pd.DataFrame(
        [
            {"defect_name": name, **segmentation_metrics_from_counts(*counts.tolist())}
            for name, counts in sorted(category_counts.items())
        ]
    )
    return {"segmentation": segmentation, "classification": classification}, per_image, categories


def _plot_threshold_search(search: pd.DataFrame, selected: float, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    for column, label in (
        ("dice", "Dice"),
        ("pixel_precision", "Precision"),
        ("pixel_recall", "Recall"),
    ):
        axis.plot(search["segmentation_threshold"], search[column], marker="o", label=label)
    axis.axvline(selected, color="black", linestyle="--", label=f"Selected: {selected:.2f}")
    axis.set(xlabel="Probability threshold", ylabel="Metric", ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(4.5, 4))
    display = axis.imshow(values, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "Defective"],
        yticklabels=["Normal", "Defective"],
        xlabel="Predicted",
        ylabel="Ground truth",
    )
    figure.colorbar(display, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _runtime(config: dict[str, object]) -> tuple[torch.device, ResNet18UNet, Path]:
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(config["checkpoint"])
    model, _ = load_model(checkpoint_path, device)
    return device, model, checkpoint_path


def select_and_freeze_thresholds(config: dict[str, object]) -> dict[str, object]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device, model, checkpoint_path = _runtime(config)
    validation = _predict_configured(config, model, "validation", device)
    segmentation_threshold, segmentation_search = select_segmentation_threshold(
        validation, [float(value) for value in config["segmentation_threshold_candidates"]]
    )
    image_threshold, image_search = select_image_threshold(validation, segmentation_threshold)
    metrics, per_image, categories = evaluate_images(
        validation, segmentation_threshold, image_threshold
    )
    frozen = {
        "selection_split": "validation",
        "segmentation_threshold": segmentation_threshold,
        "image_component_threshold": image_threshold,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_metrics": metrics,
    }
    segmentation_search.to_csv(output_dir / "segmentation_threshold_search.csv", index=False)
    image_search.to_csv(output_dir / "image_threshold_search.csv", index=False)
    per_image.to_csv(output_dir / "validation_per_image.csv", index=False)
    categories.to_csv(output_dir / "validation_by_category.csv", index=False)
    (output_dir / "frozen_thresholds.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot_threshold_search(
        segmentation_search, segmentation_threshold, output_dir / "validation_thresholds.png"
    )
    return frozen


def evaluate_frozen_test(config: dict[str, object]) -> dict[str, object]:
    output_dir = Path(config["output_dir"])
    frozen_path = output_dir / "frozen_thresholds.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Select and freeze validation thresholds before test evaluation")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    device, model, checkpoint_path = _runtime(config)
    if sha256_file(checkpoint_path) != frozen["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed after threshold selection; refusing test evaluation")
    evaluation_split = str(config.get("evaluation_split", "test"))
    test = _predict_configured(config, model, evaluation_split, device)
    metrics, per_image, categories = evaluate_images(
        test,
        float(frozen["segmentation_threshold"]),
        float(frozen["image_component_threshold"]),
    )
    result = {
        "evaluation_split": evaluation_split,
        "threshold_source": "validation",
        "segmentation_threshold": frozen["segmentation_threshold"],
        "image_component_threshold": frozen["image_component_threshold"],
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "metrics": metrics,
    }
    per_image.to_csv(output_dir / f"{evaluation_split}_per_image.csv", index=False)
    categories.to_csv(output_dir / f"{evaluation_split}_by_category.csv", index=False)
    (output_dir / f"{evaluation_split}_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot_confusion_matrix(
        metrics["classification"]["confusion_matrix"],
        output_dir / f"{evaluation_split}_confusion_matrix.png",
    )
    return result


def load_evaluation_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
