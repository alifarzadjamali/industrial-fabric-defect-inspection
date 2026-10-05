import pytest
import torch
from torch.nn import functional as F

from fabric_inspection.models.unet import ResNet18UNet
from fabric_inspection.training.losses import BCEDiceLoss, DiceLoss, FocalDiceLoss


def test_unet_output_and_combined_loss_backward() -> None:
    model = ResNet18UNet(pretrained=False)
    inputs = torch.randn(2, 3, 64, 64)
    targets = torch.zeros(2, 1, 64, 64)
    targets[:, :, 20:30, 20:30] = 1
    logits = model(inputs)
    loss = BCEDiceLoss()(logits, targets)
    loss.backward()
    assert logits.shape == targets.shape
    assert torch.isfinite(loss)


@pytest.mark.parametrize("loss", [DiceLoss, BCEDiceLoss, FocalDiceLoss])
def test_losses_reject_invalid_configuration(loss: type[torch.nn.Module]) -> None:
    with pytest.raises(ValueError):
        if loss is DiceLoss:
            loss(smooth=0)
        elif loss is BCEDiceLoss:
            loss(bce_weight=-1)
        else:
            loss(alpha=2)


def test_focal_dice_loss_matches_probability_definition() -> None:
    logits = torch.tensor([[[[-2.0, 0.5], [1.0, 3.0]]]])
    targets = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    loss = FocalDiceLoss(focal_weight=1.0, dice_weight=0.0, alpha=0.7, gamma=2.0)

    probability = torch.sigmoid(logits)
    probability_true = probability * targets + (1 - probability) * (1 - targets)
    alpha_true = 0.7 * targets + 0.3 * (1 - targets)
    expected = (
        alpha_true
        * (1 - probability_true).square()
        * F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    ).mean()

    assert torch.allclose(loss(logits, targets), expected)
