"""Binary segmentation and image-classification metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_divide(numerator: float, denominator: float, zero_value: float = 0.0) -> float:
    return numerator / denominator if denominator else zero_value


def segmentation_counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} != {target.shape}")
    true_positive = int(np.logical_and(prediction, target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())
    true_negative = int(np.logical_and(~prediction, ~target).sum())
    return true_positive, false_positive, false_negative, true_negative


def segmentation_metrics_from_counts(
    true_positive: int, false_positive: int, false_negative: int, true_negative: int
) -> dict[str, float | int]:
    dice_denominator = 2 * true_positive + false_positive + false_negative
    union = true_positive + false_positive + false_negative
    dice = safe_divide(2 * true_positive, dice_denominator, zero_value=1.0)
    return {
        "dice": dice,
        "iou": safe_divide(true_positive, union, zero_value=1.0),
        "pixel_precision": safe_divide(true_positive, true_positive + false_positive),
        "pixel_recall": safe_divide(true_positive, true_positive + false_negative),
        "pixel_f1": dice,
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "true_negative_pixels": true_negative,
    }


def classification_metrics(
    targets: Iterable[bool], predictions: Iterable[bool], scores: Iterable[float] | None = None
) -> dict[str, float | int | list[list[int]] | None]:
    target = np.asarray(list(targets), dtype=bool)
    prediction = np.asarray(list(predictions), dtype=bool)
    if target.shape != prediction.shape:
        raise ValueError("Classification targets and predictions must have equal length")
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    auc: float | None = None
    if scores is not None and len(np.unique(target)) == 2:
        auc = float(roc_auc_score(target, list(scores)))
    return {
        "accuracy": safe_divide(tp + tn, len(target)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }
