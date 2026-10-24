import torch
import pytest

from modules import BaselineUNet3D, ImprovedUNet3D


def test_baseline_3d_unet_preserves_volume_shape_and_backpropagates():
    model = BaselineUNet3D(in_channels=1, num_classes=2, base_channels=4)
    logits = model(torch.randn(1, 1, 16, 16, 16))
    loss = logits.square().mean()
    loss.backward()
    assert logits.shape == (1, 2, 16, 16, 16)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_improved_3d_unet_returns_deep_supervision_only_when_training():
    model = ImprovedUNet3D(in_channels=1, num_classes=2, base_channels=4)
    model.train()
    final_logits, auxiliaries = model(torch.randn(1, 1, 16, 16, 16))
    assert final_logits.shape == (1, 2, 16, 16, 16)
    assert len(auxiliaries) == 2
    assert all(logits.shape[1] == 2 for logits in auxiliaries)
    model.eval()
    with torch.no_grad():
        inference_logits = model(torch.randn(1, 1, 16, 16, 16))
    assert isinstance(inference_logits, torch.Tensor)
    assert inference_logits.shape == (1, 2, 16, 16, 16)


def test_improved_3d_unet_uses_dropout_and_interpolation_upsampling():
    """The revised Improved arm regularises context blocks without transposed convolutions."""
    model = ImprovedUNet3D(in_channels=1, num_classes=2, base_channels=4)
    assert any(isinstance(module, torch.nn.Dropout3d) for module in model.modules())
    assert not any(isinstance(module, torch.nn.ConvTranspose3d) for module in model.modules())


def test_improved_3d_unet_rejects_divisible_but_too_small_volume_before_normalization():
    """A 1-voxel bottleneck is invalid for instance normalization at batch size one."""
    model = ImprovedUNet3D(in_channels=1, num_classes=2, base_channels=4)

    with pytest.raises(ValueError, match="at least 16"):
        model(torch.randn(1, 1, 8, 8, 8))
