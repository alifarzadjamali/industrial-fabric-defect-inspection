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
    # The remaining pixels are true negatives; deriving this avoids one more
    # full-size boolean temporary for every segmentation map.
    true_negative = prediction.size - true_positive - false_positive - false_negative
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
    if target.size == 0:
        raise ValueError("Classification metrics require at least one sample")
    if target.shape != prediction.shape:
        raise ValueError("Classification targets and predictions must have equal length")
    score = None if scores is None else np.asarray(list(scores), dtype=float)
    if score is not None and target.shape != score.shape:
        raise ValueError("Classification targets and scores must have equal length")
    if score is not None and not np.isfinite(score).all():
        raise ValueError("Classification scores must be finite")
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = target.size - tp - fp - fn
    auc: float | None = None
    if score is not None and target.any() and not target.all():
        auc = float(roc_auc_score(target, score))
    return _classification_metrics_from_counts(tp, fp, fn, tn, auc)


def _classification_metrics_from_counts(
    tp: int, fp: int, fn: int, tn: int, auc: float | None
) -> dict[str, float | int | list[list[int]] | None]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "accuracy": safe_divide(tp + tn, tp + fp + fn + tn),
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


def classification_threshold_metrics(
    targets: Iterable[bool], scores: Iterable[float], threshold_name: str
) -> list[dict[str, float | int | list[list[int]] | None]]:
    """Evaluate every score-derived threshold using one sorted cumulative pass."""

    target = np.asarray(list(targets), dtype=bool)
    score = np.asarray(list(scores), dtype=float)
    if target.size == 0:
        raise ValueError("Classification threshold search requires at least one sample")
    if target.shape != score.shape:
        raise ValueError("Classification targets and scores must have equal length")
    if not np.isfinite(score).all():
        raise ValueError("Classification scores must be finite")

    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    positive_prefix = np.concatenate(([0], np.cumsum(target[order], dtype=np.int64)))
    candidates = np.unique(
        np.concatenate(([0.0], score, [np.nextafter(float(score.max()), np.inf)]))
    )
    split_indices = np.searchsorted(sorted_score, candidates, side="left")
    positive_total = int(positive_prefix[-1])
    true_positive = positive_total - positive_prefix[split_indices]
    predicted_positive = len(score) - split_indices
    false_positive = predicted_positive - true_positive
    false_negative = positive_total - true_positive
    true_negative = split_indices - positive_prefix[split_indices]
    auc = (
        float(roc_auc_score(target, score)) if positive_total not in {0, len(target)} else None
    )
    return [
        {
            threshold_name: float(threshold),
            **_classification_metrics_from_counts(
                int(true_positive[index]),
                int(false_positive[index]),
                int(false_negative[index]),
                int(true_negative[index]),
                auc,
            ),
        }
        for index, threshold in enumerate(candidates)
    ]
