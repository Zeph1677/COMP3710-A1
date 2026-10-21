from pathlib import Path

import pytest
import torch

from compare_failures import build_selection_manifest, common_error_score, error_relevant_prostate_slice, rank_failure_cases, render_comparative_prostate_panel, validate_comparison_inputs


def _metrics(**overrides):
    value = {
        "manifest_sha256": "manifest",
        "prostate_label": 5,
        "prostate_label_source": "CSIRO source",
        "target_shape": [128, 128, 128],
        "patch_size": [128, 128, 128],
        "stride": [64, 64, 64],
        "test_patient_count": 1,
        "per_patient_prostate_dice": {"B006": 0.5},
    }
    value.update(overrides)
    return value


def test_common_error_score_uses_both_models_voxel_errors():
    assert common_error_score({"false_positive_voxels": 3, "false_negative_voxels": 7}, {"false_positive_voxels": 5, "false_negative_voxels": 1}) == 8.0


def test_error_relevant_slice_is_deterministic_inside_prostate_extent():
    target = torch.zeros((5, 3, 3), dtype=torch.long)
    baseline = target.clone()
    improved = target.clone()
    target[1, 1, 1] = 5
    target[3, 1, 1] = 5
    baseline[3, 1, 1] = 0
    improved[3, 1, 1] = 0

    assert error_relevant_prostate_slice(target, baseline, improved, prostate_label=5) == 3


def test_comparison_rejects_different_manifest_or_test_patients():
    with pytest.raises(ValueError, match="manifest_sha256"):
        validate_comparison_inputs(_metrics(), _metrics(manifest_sha256="other"))
    with pytest.raises(ValueError, match="test patient"):
        validate_comparison_inputs(_metrics(), _metrics(per_patient_prostate_dice={"C007": 0.5}))


def _prediction(case_id: str, patient_id: str = "B006", manifest_sha256: str = "manifest", prostate_label: int = 5):
    target = torch.zeros((3, 3, 3), dtype=torch.long)
    target[1, 1, 1] = prostate_label
    return {"case_id": case_id, "patient_id": patient_id, "manifest_sha256": manifest_sha256, "prostate_label": prostate_label, "target_shape": [128, 128, 128], "image": torch.zeros((1, 3, 3, 3)), "target": target, "prediction": target.clone()}


def test_failure_ranking_rejects_missing_or_extra_prediction_cases():
    with pytest.raises(ValueError, match="prediction case sets"):
        rank_failure_cases({"B006_Week0": _prediction("B006_Week0")}, {"C007_Week0": _prediction("C007_Week0", patient_id="C007")}, prostate_label=5)


def test_failure_ranking_rejects_prediction_metadata_mismatch():
    with pytest.raises(ValueError, match="manifest_sha256"):
        rank_failure_cases({"B006_Week0": _prediction("B006_Week0")}, {"B006_Week0": _prediction("B006_Week0", manifest_sha256="other")}, prostate_label=5)


def test_comparison_panel_renders_the_deterministic_error_relevant_slice(tmp_path: Path):
    image = torch.zeros((4, 4, 4), dtype=torch.float32)
    target = torch.zeros((4, 4, 4), dtype=torch.long)
    baseline = target.clone()
    improved = target.clone()
    target[2, 1:3, 1:3] = 5
    baseline[2, 1:3, 1:3] = 0
    output = tmp_path / "comparison.png"

    slice_index = render_comparative_prostate_panel(image, target, baseline, improved, prostate_label=5, output_path=output)

    assert slice_index == 2
    assert output.is_file() and output.stat().st_size > 0


def test_selection_manifest_records_rule_and_frozen_git_sha():
    manifest = build_selection_manifest([{"patient_id": "B006", "case_id": "B006_Week0", "slice_index": 4, "error_score": 7.0}], git_sha="0950a0f")

    assert manifest["git_sha"] == "0950a0f"
    assert manifest["rule_version"] == 1
    assert manifest["selected_cases"][0]["patient_id"] == "B006"
