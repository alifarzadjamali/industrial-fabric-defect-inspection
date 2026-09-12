"""Create the fixed README example figure from the frozen enhanced model."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fabric_inspection.evaluation.model_evaluator import load_model, predict_source_images

CHECKPOINT = Path("checkpoints/enhanced_v2_seed42/best.pt")
MANIFEST = Path("data/processed/enhanced_splits.csv")
METRICS = Path("results/evaluation/enhanced_v2_seed42/final_test_per_image.csv")
OUTPUT = Path("docs/figures/enhanced_inspection_examples.png")
EXAMPLES = ("0102_010_03", "0010_006_02")
SEGMENTATION_THRESHOLD = 0.03


def _crop_bounds(mask: np.ndarray, width: int = 896) -> tuple[int, int]:
    columns = np.flatnonzero(mask.any(axis=0))
    centre = int(round(float(columns.mean()))) if len(columns) else mask.shape[1] // 2
    start = min(max(centre - width // 2, 0), max(mask.shape[1] - width, 0))
    return start, min(start + width, mask.shape[1])


def _overlay(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    rgb = np.repeat((image.astype(np.float32) / 255.0)[..., None], 3, axis=2)
    rgb[prediction] = 0.55 * rgb[prediction] + 0.45 * np.array([1.0, 0.1, 0.35])
    contours, _ = cv2.findContours(
        target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(rgb, contours, -1, (0.2, 1.0, 0.35), thickness=2)
    return np.clip(rgb, 0.0, 1.0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(CHECKPOINT, device)
    all_predictions = predict_source_images(
        model,
        MANIFEST,
        "final_test",
        image_size=256,
        tile_size=128,
        overlap=64,
        batch_size=24,
        device=device,
        mixed_precision=True,
    )
    predictions = {item.image_id: item for item in all_predictions}
    metrics = pd.read_csv(METRICS).set_index("image_id")

    figure, axes = plt.subplots(2, 4, figsize=(16, 5.4), constrained_layout=True)
    column_titles = ("Fabric image", "Ground-truth mask", "Predicted mask", "Overlay")
    for axis, title in zip(axes[0], column_titles, strict=True):
        axis.set_title(title, fontsize=12, weight="semibold", pad=8)

    for row_index, image_id in enumerate(EXAMPLES):
        item = predictions[image_id]
        prediction = item.probability >= SEGMENTATION_THRESHOLD
        with_path = pd.read_csv(MANIFEST).set_index("image_id").loc[image_id, "image_path"]
        image = cv2.imread(str(with_path), cv2.IMREAD_GRAYSCALE)
        start, end = _crop_bounds(item.target)
        source_crop = image[:, start:end]
        target_crop = item.target[:, start:end]
        prediction_crop = prediction[:, start:end]
        panels = (
            (source_crop, "gray", 0, 255),
            (target_crop, "gray", 0, 1),
            (prediction_crop, "gray", 0, 1),
            (_overlay(source_crop, target_crop, prediction_crop), None, None, None),
        )
        for axis, (panel, colour_map, minimum, maximum) in zip(
            axes[row_index], panels, strict=True
        ):
            axis.imshow(panel, cmap=colour_map, vmin=minimum, vmax=maximum, aspect="auto")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#d0d7de")
        result = metrics.loc[image_id]
        outcome = "Detected" if bool(result["predicted_defective"]) else "Missed"
        axes[row_index, 0].set_ylabel(
            f"{outcome}: {item.defect_name.replace('_', ' ')}\nDice {result['dice']:.3f}",
            fontsize=10,
            rotation=90,
            labelpad=12,
        )

    figure.suptitle("Enhanced model: representative final-holdout inspections", fontsize=15)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
