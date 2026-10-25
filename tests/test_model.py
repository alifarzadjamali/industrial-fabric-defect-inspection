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


def test_unet_preserves_odd_input_shape() -> None:
    model = ResNet18UNet(pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 65, 67))
    assert output.shape == (1, 1, 65, 67)


def test_dice_loss_rejects_mismatched_and_empty_inputs() -> None:
    loss = DiceLoss()
    with pytest.raises(ValueError, match="equal shapes"):
        loss(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 1))
    with pytest.raises(ValueError, match="at least one pixel"):
        loss(torch.empty(0, 1, 2, 2), torch.empty(0, 1, 2, 2))


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


def test_dice_half_precision_large_masks_have_finite_gradients() -> None:
    logits = torch.full((1, 1, 256, 256), 2.0, dtype=torch.float16, requires_grad=True)
    targets = torch.ones_like(logits)
    loss = DiceLoss()(logits, targets)
    reference = DiceLoss()(logits.detach().float(), targets.float())
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
