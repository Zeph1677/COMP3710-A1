import hashlib
import json
from pathlib import Path

import pytest

from formal_wide20_protocol import validate_final_sacct, verify_wide20_checkpoint


GIT_SHA = "2e2529857a586a009f0f4cad2e260dc810531ac0"
MANIFEST_SHA = "manifest-sha"
SOURCE = "CSIRO, Labelled weekly MR images of the male pelvis, v2, DOI: 10.25919/45T8-P065, BriefDataDescription.txt, Labels block: 5=Prostate."


def _write_wide20_run(tmp_path: Path) -> tuple[Path, str]:
    run_dir = tmp_path / "candidate_05_wide20"
    training = run_dir / "training_improved"
    training.mkdir(parents=True)
    checkpoint = training / "best.pt"
    checkpoint.write_bytes(b"frozen checkpoint bytes")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (run_dir / "git_sha.txt").write_text(GIT_SHA + "\n", encoding="utf-8")
    (run_dir / "re_audited_manifest.json").write_text("{}\n", encoding="utf-8")
    (training / "experiment_config.json").write_text(json.dumps({
        "manifest_sha256": MANIFEST_SHA,
        "model": "improved",
        "base_channels": 20,
        "learning_rate": 0.001,
        "prostate_ce_weight": 4.0,
        "prostate_dice_weight": 4.0,
        "foreground_patch_probability": 0.5,
        "epochs": 20,
        "selection_metric": "prostate_dice",
        "seed": 42,
        "prostate_label": 5,
        "prostate_label_source": SOURCE,
        "target_shape": [128, 128, 128],
        "train_patch_shape": [96, 96, 96],
    }) + "\n", encoding="utf-8")
    (training / "history.json").write_text(json.dumps([{"epoch": 11}]) + "\n", encoding="utf-8")
    (training / "training_summary.json").write_text(json.dumps({"selected_epoch": 11, "selection_metric": "prostate_dice"}) + "\n", encoding="utf-8")
    return run_dir, digest


def test_verify_wide20_checkpoint_accepts_complete_frozen_candidate(tmp_path: Path):
    run_dir, digest = _write_wide20_run(tmp_path)

    report = verify_wide20_checkpoint(run_dir, expected_git_sha=GIT_SHA, expected_manifest_sha256=MANIFEST_SHA)

    assert report["checkpoint_sha256"] == digest
    assert report["selected_epoch"] == 11
    assert report["selection_metric"] == "prostate_dice"
    assert report["route"] == "reuse_frozen_checkpoint"


def test_verify_wide20_checkpoint_rejects_changed_checkpoint_bytes(tmp_path: Path):
    run_dir, digest = _write_wide20_run(tmp_path)
    (run_dir / "training_improved" / "best.pt").write_bytes(b"changed bytes")

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        verify_wide20_checkpoint(run_dir, expected_git_sha=GIT_SHA, expected_manifest_sha256=MANIFEST_SHA, expected_checkpoint_sha256=digest)


@pytest.mark.parametrize("field,value", [("base_channels", 16), ("selection_metric", "mean_dice"), ("prostate_label", 4)])
def test_verify_wide20_checkpoint_rejects_protocol_mismatch(tmp_path: Path, field: str, value: object):
    run_dir, _ = _write_wide20_run(tmp_path)
    config_path = run_dir / "training_improved" / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        verify_wide20_checkpoint(run_dir, expected_git_sha=GIT_SHA, expected_manifest_sha256=MANIFEST_SHA)


def test_validate_final_sacct_accepts_completed_zero_exit_row():
    record = validate_final_sacct("683999|hipmri-formal|COMPLETED|0:0|01:02:03|cpu=4,gpu=1\n", job_id="683999")

    assert record == {"JobID": "683999", "JobName": "hipmri-formal", "State": "COMPLETED", "ExitCode": "0:0", "Elapsed": "01:02:03", "AllocTRES": "cpu=4,gpu=1"}


@pytest.mark.parametrize("text", [
    "683999|hipmri-formal|RUNNING|0:0|00:10:00|cpu=4,gpu=1\n",
    "683999|hipmri-formal|FAILED|1:0|00:10:00|cpu=4,gpu=1\n",
])
def test_validate_final_sacct_rejects_nonfinal_or_failed_accounting(text: str):
    with pytest.raises(ValueError, match="final accounting"):
        validate_final_sacct(text, job_id="683999")
