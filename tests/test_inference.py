import numpy as np
import pytest
import torch
from torch import nn

from fabric_inspection.inference.tiling import predict_grayscale_image


class ZeroLogitModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            inputs.shape[0], 1, inputs.shape[2], inputs.shape[3], device=inputs.device
        )


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
