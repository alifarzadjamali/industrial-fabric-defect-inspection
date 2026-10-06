import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fabric_inspection.training.trainer import _run_epoch


def test_epoch_metrics_derive_errors_from_positive_counts() -> None:
    logits = torch.tensor([[[[8.0, -8.0], [8.0, -8.0]]]])
    targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    loader = DataLoader(TensorDataset(logits, targets), batch_size=1)
    metrics = _run_epoch(
        nn.Identity(),
        loader,
        nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
        metric_threshold=0.5,
        progress_bar=False,
    )
    assert metrics["dice"] == 0.5
    assert metrics["iou"] == 1 / 3


def test_epoch_metric_threshold_is_applied_in_logit_space() -> None:
    logits = torch.tensor([[[[1.0, 2.0]]]])
    targets = torch.tensor([[[[0.0, 1.0]]]])
    loader = DataLoader(TensorDataset(logits, targets), batch_size=1)
    metrics = _run_epoch(
        nn.Identity(),
        loader,
        nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
        metric_threshold=0.8,
        progress_bar=False,
    )
    assert metrics["dice"] == 1.0


@pytest.mark.parametrize("threshold", [0.0, 1.0, float("nan")])
def test_epoch_rejects_invalid_metric_thresholds(threshold: float) -> None:
    loader = DataLoader(TensorDataset(torch.zeros(1, 1), torch.zeros(1, 1)))
    with pytest.raises(ValueError, match="Metric threshold"):
        _run_epoch(
            nn.Identity(),
            loader,
            nn.BCEWithLogitsLoss(),
            torch.device("cpu"),
            metric_threshold=threshold,
            progress_bar=False,
        )
