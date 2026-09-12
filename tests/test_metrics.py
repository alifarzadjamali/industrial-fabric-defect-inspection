import numpy as np

from fabric_inspection.evaluation.metrics import (
    classification_metrics,
    segmentation_counts,
    segmentation_metrics_from_counts,
)


def test_segmentation_metrics_have_expected_values() -> None:
    target = np.array([[1, 1], [0, 0]], dtype=bool)
    prediction = np.array([[1, 0], [1, 0]], dtype=bool)
    counts = segmentation_counts(prediction, target)
    metrics = segmentation_metrics_from_counts(*counts)
    assert counts == (1, 1, 1, 1)
    assert metrics["dice"] == 0.5
    assert metrics["iou"] == 1 / 3
    assert metrics["pixel_precision"] == 0.5
    assert metrics["pixel_recall"] == 0.5


def test_classification_metrics_prioritise_recall_visibility() -> None:
    metrics = classification_metrics(
        targets=[False, False, True, True],
        predictions=[False, True, True, False],
        scores=[0.1, 0.8, 0.9, 0.2],
    )
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["accuracy"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["roc_auc"] == 0.75
