import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_report_results import build_report_summary, ensure_output_is_not_source


SOURCE = "CSIRO, Labelled weekly MR images of the male pelvis, v2, DOI: 10.25919/45T8-P065, BriefDataDescription.txt, Labels block: 5=Prostate."


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_arm(root: Path, name: str, patients: dict[str, float], *, manifest: str = "manifest", model: str = "baseline", selection_metric: str = "mean_dice", include_resources: bool = True) -> dict[str, str]:
    arm = root / name
    _write_json(arm / "training" / "experiment_config.json", {
        "manifest_sha256": manifest,
        "model": model,
        "prostate_label": 5,
        "prostate_label_source": SOURCE,
        "selection_metric": selection_metric,
    })
    _write_json(arm / "training" / "training_summary.json", {
        "selected_epoch": 3,
        "parameter_count": 100,
        "peak_train_vram_mb": 10.0,
    })
    metrics = {
        "manifest_sha256": manifest,
        "prostate_label": 5,
        "prostate_label_source": SOURCE,
        "per_patient_prostate_dice": patients,
        "mean_patient_prostate_dice": sum(patients.values()) / len(patients),
        "patch_size": [128, 128, 128],
        "stride": [64, 64, 64],
        "logit_merge_method": "overlap_averaged_logits",
    }
    if include_resources:
        metrics.update({"peak_inference_vram_mb": 12.0, "mean_patient_inference_seconds": 0.5})
    _write_json(arm / "test" / "test_metrics.json", metrics)
    (arm / "git_sha.txt").write_text("deadbeef\n", encoding="utf-8")
    return {
        "config": str((arm / "training" / "experiment_config.json").relative_to(root)),
        "training_summary": str((arm / "training" / "training_summary.json").relative_to(root)),
        "metrics": str((arm / "test" / "test_metrics.json").relative_to(root)),
        "git_sha": str((arm / "git_sha.txt").relative_to(root)),
    }


def test_build_report_summary_preserves_patient_means_pairwise_deltas_and_missing_fields(tmp_path: Path):
    baseline = _write_arm(tmp_path, "baseline", {"A": 0.8, "B": 0.6}, model="baseline", include_resources=False)
    improved = _write_arm(tmp_path, "improved", {"A": 0.7, "B": 0.9}, model="improved")

    summary = build_report_summary(tmp_path, [{
        "group_id": "fixture_group",
        "selection_metric": "mean_dice",
        "baseline": baseline,
        "improved": improved,
    }])

    group = summary["experiment_groups"][0]
    assert group["headline"]["baseline_mean_patient_prostate_dice"] == pytest.approx(0.7)
    assert group["headline"]["improved_mean_patient_prostate_dice"] == pytest.approx(0.8)
    assert group["headline"]["improved_minus_baseline"] == pytest.approx(0.1)
    assert [row["patient_id"] for row in group["paired_patient_dice"]] == ["A", "B"]
    assert group["paired_patient_dice"][0]["improved_minus_baseline"] == pytest.approx(-0.1)
    assert group["paired_patient_dice"][1]["improved_minus_baseline"] == pytest.approx(0.3)
    assert "mean_patient_inference_seconds" in group["arms"]["baseline"]["missing_fields"]
    assert group["arms"]["baseline"]["source_files"]["metrics"]["sha256"]


def test_build_report_summary_rejects_protocol_or_patient_set_mismatch(tmp_path: Path):
    baseline = _write_arm(tmp_path, "baseline", {"A": 0.8, "B": 0.6}, manifest="one", model="baseline")
    improved = _write_arm(tmp_path, "improved", {"A": 0.7, "C": 0.9}, manifest="two", model="improved")

    with pytest.raises(ValueError, match="manifest_sha256|patient_ids"):
        build_report_summary(tmp_path, [{
            "group_id": "mismatch",
            "selection_metric": "mean_dice",
            "baseline": baseline,
            "improved": improved,
        }])


def test_summary_output_cannot_overwrite_a_declared_source_file(tmp_path: Path):
    baseline = _write_arm(tmp_path, "baseline", {"A": 0.8, "B": 0.6}, model="baseline")
    improved = _write_arm(tmp_path, "improved", {"A": 0.7, "B": 0.9}, model="improved")
    groups = [{
        "group_id": "fixture_group",
        "selection_metric": "mean_dice",
        "baseline": baseline,
        "improved": improved,
    }]

    with pytest.raises(ValueError, match="refuses to overwrite declared source file"):
        ensure_output_is_not_source(tmp_path, Path(baseline["metrics"]), groups)
