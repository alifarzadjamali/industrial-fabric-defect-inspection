import torch

from fabric_inspection.models.unet import ResNet18UNet
from fabric_inspection.training.losses import BCEDiceLoss


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
