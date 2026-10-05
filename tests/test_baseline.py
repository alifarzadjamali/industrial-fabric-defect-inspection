import numpy as np
import pytest

from fabric_inspection.baseline.local_texture import anomaly_score, postprocess_mask


def test_local_anomaly_is_detected_and_small_response_is_removed() -> None:
    image = np.full((64, 128), 120, dtype=np.uint8)
    image[24:40, 56:72] = 230
    score = anomaly_score(image, gaussian_sigma=7, local_sigma=2)
    prediction = postprocess_mask(score, threshold=2, morphology_kernel=3, min_component_area=20)
    assert score.shape == image.shape
    assert prediction[30, 64]


def test_postprocessing_filters_components_by_area() -> None:
    score = np.zeros((12, 12), dtype=np.float32)
    score[1:3, 1:3] = 1.0
    score[6:9, 6:9] = 1.0
    prediction = postprocess_mask(
        score, threshold=0.5, morphology_kernel=1, min_component_area=5
    )
    assert not prediction[1:3, 1:3].any()
    assert prediction[6:9, 6:9].all()
    assert int(prediction.sum()) == 9


def test_postprocessing_returns_boolean_mask_without_component_filtering() -> None:
    score = np.array([[0.1, 0.7]], dtype=np.float32)
    prediction = postprocess_mask(
        score, threshold=0.5, morphology_kernel=1, min_component_area=1
    )
    assert prediction.dtype == np.bool_
    assert prediction.tolist() == [[False, True]]


@pytest.mark.parametrize("argument", ["gaussian_sigma", "local_sigma"])
def test_anomaly_score_rejects_non_positive_sigma(argument: str) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        anomaly_score(np.ones((8, 8), dtype=np.uint8), **{argument: 0})


@pytest.mark.parametrize("argument", ["morphology_kernel", "min_component_area"])
def test_postprocessing_rejects_non_positive_sizes(argument: str) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        postprocess_mask(
            np.ones((8, 8), dtype=np.float32), threshold=0.5, **{argument: 0}
        )
