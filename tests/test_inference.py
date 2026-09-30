import numpy as np
import pytest
import torch
from torch import nn

from fabric_inspection.inference.tiling import _blend_window, predict_grayscale_image


class ZeroLogitModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            inputs.shape[0], 1, inputs.shape[2], inputs.shape[3], device=inputs.device
        )


class BatchRecordingModel(ZeroLogitModel):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(len(inputs))
        return super().forward(inputs)


def test_tiled_inference_preserves_non_multiple_image_shape() -> None:
    image = np.full((32, 50), 127, dtype=np.uint8)
    probability = predict_grayscale_image(
        ZeroLogitModel(),
        image,
        torch.device("cpu"),
        image_size=32,
        batch_size=2,
        mixed_precision=False,
    )
    assert probability.shape == image.shape
    assert np.allclose(probability, 0.5)


def test_tiled_inference_handles_single_pixel_images() -> None:
    image = np.array([[127]], dtype=np.uint8)
    probability = predict_grayscale_image(
        ZeroLogitModel(), image, torch.device("cpu"), image_size=8, mixed_precision=False
    )
    assert probability.shape == image.shape
    assert np.allclose(probability, 0.5)


def test_tiled_inference_rejects_empty_images() -> None:
    with pytest.raises(ValueError, match="at least one pixel"):
        predict_grayscale_image(
            ZeroLogitModel(),
            np.empty((0, 8), dtype=np.uint8),
            torch.device("cpu"),
            mixed_precision=False,
        )


def test_overlapping_scaled_inference_preserves_image_shape() -> None:
    image = np.full((51, 73), 127, dtype=np.uint8)
    probability = predict_grayscale_image(
        ZeroLogitModel(),
        image,
        torch.device("cpu"),
        image_size=32,
        tile_size=16,
        overlap=8,
        batch_size=4,
        mixed_precision=False,
    )
    assert probability.shape == image.shape
    assert np.allclose(probability, 0.5, atol=1e-6)


def test_tiled_inference_streams_only_one_batch_at_a_time() -> None:
    image = np.full((32, 80), 127, dtype=np.uint8)
    model = BatchRecordingModel()
    predict_grayscale_image(
        model,
        image,
        torch.device("cpu"),
        image_size=16,
        batch_size=3,
        mixed_precision=False,
    )
    assert model.batch_sizes == [3, 3, 3, 1]


def test_blend_windows_are_reused_and_read_only() -> None:
    _blend_window.cache_clear()
    first = _blend_window(16, 8)
    second = _blend_window(16, 8)
    assert first is second
    assert not first.flags.writeable
    assert _blend_window.cache_info().hits == 1


@pytest.mark.parametrize(
    ("argument", "value"),
    [("image_size", 0), ("tile_size", 0), ("batch_size", 0)],
)
def test_tiled_inference_rejects_non_positive_sizes(argument: str, value: int) -> None:
    image = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="must be positive"):
        predict_grayscale_image(
            ZeroLogitModel(),
            image,
            torch.device("cpu"),
            mixed_precision=False,
            **{argument: value},
        )
