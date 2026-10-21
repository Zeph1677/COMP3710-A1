from experiment_configs import load_candidate_configuration


def test_candidate_configurations_are_explicit_and_share_fixed_geometry():
    preflight = load_candidate_configuration("preflight-a")
    formal = load_candidate_configuration("formal-b")

    assert preflight["target_shape"] == formal["target_shape"] == [128, 128, 128]
    assert preflight["train_patch_shape"] == formal["train_patch_shape"] == [96, 96, 96]
    assert preflight["prostate_ce_weight"] == preflight["prostate_dice_weight"] == 4.0
    assert formal["prostate_ce_weight"] == formal["prostate_dice_weight"] == 8.0
    assert preflight["foreground_patch_probability"] == 0.5
    assert formal["foreground_patch_probability"] == 0.75
    assert "model" not in preflight
    assert "model" not in formal
