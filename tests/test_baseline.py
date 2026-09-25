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
