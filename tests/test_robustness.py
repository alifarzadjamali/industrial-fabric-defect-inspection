import numpy as np

from fabric_inspection.evaluation.analysis import _transform

ROBUSTNESS = {
    "blur_sigma": 1.2,
    "noise_standard_deviation": 5.0,
    "resize_factor": 0.75,
}


def test_photometric_transform_does_not_change_mask() -> None:
    image = np.full((32, 64), 100, dtype=np.uint8)
    target = np.zeros_like(image, dtype=bool)
    target[10:15, 20:25] = True
    transformed, transformed_target = _transform(
        "brightness_1.15", image, target, ROBUSTNESS, np.random.default_rng(42)
    )
    assert transformed.mean() == 115
    assert np.array_equal(transformed_target, target)


def test_rotation_transforms_image_and_mask_together() -> None:
    image = np.zeros((32, 64), dtype=np.uint8)
    target = np.zeros_like(image, dtype=bool)
    image[10:15, 20:25] = 255
    target[10:15, 20:25] = True
    transformed, transformed_target = _transform(
        "rotation_3.0", image, target, ROBUSTNESS, np.random.default_rng(42)
    )
    assert transformed.shape == image.shape
    assert transformed_target.shape == target.shape
    assert transformed_target.any()
