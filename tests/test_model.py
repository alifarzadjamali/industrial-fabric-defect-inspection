import pytest
import torch

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
