import numpy as np

from fabric_inspection.evaluation.model_evaluator import (
    ImagePrediction,
    largest_component_fraction,
    select_segmentation_threshold,
)


def test_threshold_selection_uses_dice() -> None:
    probability = np.array([[0.9, 0.6], [0.4, 0.1]], dtype=np.float32)
    target = np.array([[1, 1], [0, 0]], dtype=bool)
    image = ImagePrediction("sample", "defect", True, probability, target)
    threshold, search = select_segmentation_threshold([image], [0.3, 0.5, 0.7])
    assert threshold == 0.5
    assert float(search.loc[search["segmentation_threshold"] == 0.5, "dice"].iloc[0]) == 1.0


def test_largest_component_ignores_disconnected_total_area() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[6:9, 6:9] = True
    assert largest_component_fraction(mask) == 0.09
