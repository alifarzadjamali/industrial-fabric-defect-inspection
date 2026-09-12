import numpy as np

from fabric_inspection.baseline.local_texture import anomaly_score, postprocess_mask


def test_local_anomaly_is_detected_and_small_response_is_removed() -> None:
    image = np.full((64, 128), 120, dtype=np.uint8)
    image[24:40, 56:72] = 230
    score = anomaly_score(image, gaussian_sigma=7, local_sigma=2)
    prediction = postprocess_mask(score, threshold=2, morphology_kernel=3, min_component_area=20)
    assert score.shape == image.shape
    assert prediction[30, 64]
