"""Leakage-safe full-image model evaluation and threshold selection."""

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
from PIL import Image
from torch.utils.data import DataLoader

from fabric_inspection.data.dataset import AitexPatchDataset, _seed_worker
from fabric_inspection.evaluation.metrics import (
    classification_metrics,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    metrics["macro_defective_dice"] = float(np.mean(defective_dice))
    return metrics


def select_segmentation_threshold(
    images: list[ImagePrediction], candidates: list[float]
) -> tuple[float, pd.DataFrame]:
    rows = [
        {"segmentation_threshold": threshold, **_segmentation_summary(images, threshold)}
        for threshold in candidates
    ]
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
    candidates = sorted({0.0, *scores, float(np.nextafter(max(scores), np.inf))})
    rows: list[dict[str, object]] = []
    for threshold in candidates:
        metrics = classification_metrics(targets, [score >= threshold for score in scores], scores)
        rows.append({"image_component_threshold": threshold, **metrics})
    search = pd.DataFrame(rows)
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
    for image in images:
        prediction = image.probability >= segmentation_threshold
        counts = segmentation_counts(prediction, image.target)
        metrics = segmentation_metrics_from_counts(*counts)
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
        category_counts.setdefault(image.defect_name, np.zeros(4, dtype=np.int64))
        category_counts[image.defect_name] += counts
    per_image = pd.DataFrame(rows)
    segmentation = _segmentation_summary(images, segmentation_threshold)
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
        "checkpoint_sha256": _sha256(checkpoint_path),
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
    if _sha256(checkpoint_path) != frozen["checkpoint_sha256"]:
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
    per_image.to_csv(output_dir / "test_per_image.csv", index=False)
    categories.to_csv(output_dir / "test_by_category.csv", index=False)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot_confusion_matrix(
        metrics["classification"]["confusion_matrix"], output_dir / "test_confusion_matrix.png"
    )
    return result


def load_evaluation_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
