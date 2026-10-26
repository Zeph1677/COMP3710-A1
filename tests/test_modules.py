import torch
import pytest

from modules import BaselineUNet3D, ContextImprovedUNet3D, ImprovedUNet3D


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
    model = ContextImprovedUNet3D(in_channels=1, num_classes=2, base_channels=4)
    assert any(isinstance(module, torch.nn.Dropout3d) for module in model.modules())
    assert not any(isinstance(module, torch.nn.ConvTranspose3d) for module in model.modules())


def test_improved_3d_unet_rejects_divisible_but_too_small_volume_before_normalization():
    """A 1-voxel bottleneck is invalid for instance normalization at batch size one."""
    model = ContextImprovedUNet3D(in_channels=1, num_classes=2, base_channels=4)

    with pytest.raises(ValueError, match="at least 16"):
        model(torch.randn(1, 1, 8, 8, 8))

@pytest.mark.parametrize('version', [None, 'improved_v1', 'legacy_improved_v1', 'improved_context_v2'])
def test_versioned_improved_round_trip_preserves_logits(tmp_path, version):
    from modules import build_model
    from predict import load_checkpoint_model
    config = {'model': 'improved', 'num_classes': 3, 'base_channels': 2, 'dropout_probability': 0.47}
    if version is not None:
        config['architecture_version'] = version
    torch.manual_seed(7)
    original = build_model(config).eval()
    if version != 'improved_context_v2':
        assert isinstance(original, ImprovedUNet3D)
    else:
        assert isinstance(original, ContextImprovedUNet3D)
        assert all(m.p == 0.47 for m in original.modules() if isinstance(m, torch.nn.Dropout3d))
    value = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        expected = original(value)
        torch.testing.assert_close(expected, original(value))
    checkpoint = tmp_path / 'roundtrip.pt'
    torch.save({'config': config, 'model_state_dict': original.state_dict()}, checkpoint)
    restored, _ = load_checkpoint_model(checkpoint, torch.device('cpu'))
    with torch.no_grad():
        torch.testing.assert_close(restored(value), expected)
    original.train()
    final, auxiliary = original(value)
    assert final.shape == expected.shape and len(auxiliary) == 2


def test_improved_v1_parameter_count_matches_main_experiment():
    from modules import build_model
    model = build_model({'model':'improved','architecture_version':'improved_v1','num_classes':6,'base_channels':16})
    assert sum(p.numel() for p in model.parameters()) == 1423106


def test_factory_rejects_unknown_architecture_and_invalid_shape_settings():
    from modules import build_model
    for config in [
        {'model':'improved','architecture_version':'unknown','num_classes':2,'base_channels':2},
        {'model':'baseline','num_classes':2,'base_channels':0},
        {'model':'baseline','num_classes':True,'base_channels':2},
    ]:
        with pytest.raises(ValueError):
            build_model(config)


@pytest.mark.parametrize('damage', ['missing_key','wrong_shape'])
def test_checkpoint_loading_is_strict(tmp_path, damage):
    from modules import build_model
    from predict import load_checkpoint_model
    config={'model':'improved','num_classes':2,'base_channels':2}
    state=build_model(config).state_dict()
    key=next(iter(state))
    if damage=='missing_key':
        del state[key]
    else:
        state[key]=torch.empty(1)
    checkpoint=tmp_path/'broken.pt'
    torch.save({'config':config,'model_state_dict':state},checkpoint)
    with pytest.raises(RuntimeError):
        load_checkpoint_model(checkpoint,torch.device('cpu'))
