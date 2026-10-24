import shutil
import subprocess
from pathlib import Path

import pytest


def test_rangpur_smoke_script_records_declared_mapping_and_reproducibility_evidence():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_hipmri_smoke.slurm").read_text(encoding="utf-8")

    assert "#SBATCH --account=comp3710" in script
    assert "#SBATCH --partition=comp3710" in script
    assert "#SBATCH --mem=" not in script
    assert 'export PATH="/opt/slurm/bin:$PATH"' in script
    assert "/home/groups/comp3710/HipMRI_Study_open" in script
    assert "--prostate-label 5" in script
    assert "10.25919/45T8-P065" in script
    assert "EXPECTED_SOURCE_NORMALIZED_SHA256=\"89C5EF8336573B6C359D0828E2FFCF6819149490FEE20071F7436ED411DD89AF\"" in script
    assert "git rev-parse HEAD" in script
    assert "nvidia-smi" in script
    assert "python src/audit.py" in script
    assert "python src/train.py" in script
    assert "python src/predict.py" in script
    assert 'CONFIG_CANDIDATE="${HIPMRI_CONFIG_CANDIDATE:-preflight-a}"' in script
    assert 'python src/experiment_configs.py --candidate "$CONFIG_CANDIDATE" --shell' in script
    assert "--train-patch-shape $CONFIG_TRAIN_PATCH_SHAPE" in script
    assert "--foreground-patch-probability $CONFIG_FOREGROUND_PATCH_PROBABILITY" in script
    assert "python train.py" not in script
    assert "python predict.py" not in script
    assert 'MODEL_NAME="${HIPMRI_MODEL:-baseline}"' in script
    assert 'RUN_DIR="$PROJECT_DIR/artifacts/${JOB_ID}_formal_${MODEL_NAME}"' in script
    assert script.index('MODEL_NAME="${HIPMRI_MODEL:-baseline}"') < script.index('RUN_DIR="$PROJECT_DIR/artifacts/${JOB_ID}_formal_${MODEL_NAME}"')
    assert "HIPMRI_MODEL must be baseline or improved" in script
    assert "set -euo pipefail" in script
    assert 'HIPMRI_EPOCHS="${HIPMRI_EPOCHS:-20}"' in script
    assert 'HIPMRI_RUN_KIND="${HIPMRI_RUN_KIND:-formal}"' in script
    assert 'sacct -P -j "$JOB_ID"' in script
    assert "State,ExitCode,Elapsed,AllocTRES" in script
    assert "--preflight-patient-id" not in script


def test_preflight_script_uses_scheduler_relative_paths_and_requires_train_patient():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_hipmri_preflight.slurm").read_text(encoding="utf-8")

    assert 'PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit this file with sbatch from the project directory}"' in script
    assert 'RUN_DIR="$PROJECT_DIR/artifacts/${JOB_ID}_preflight"' in script
    assert 'TRAINING_DIR="$RUN_DIR/training"' in script
    assert '--output-dir "$TRAINING_DIR"' in script
    assert "python src/train.py" in script
    assert "python train.py" not in script
    assert 'source "$HOME/miniconda3/bin/activate"' in script
    assert 'HIPMRI_PREFLIGHT_PATIENT_ID="${HIPMRI_PREFLIGHT_PATIENT_ID:?set a manifest train patient ID}"' in script
    assert 'HIPMRI_PREFLIGHT_UPDATES="${HIPMRI_PREFLIGHT_UPDATES:-200}"' in script
    assert 'HIPMRI_PREFLIGHT_EVAL_EVERY="${HIPMRI_PREFLIGHT_EVAL_EVERY:-20}"' in script
    assert "#SBATCH --time=03:00:00" in script
    assert 'HIPMRI_PREFLIGHT_MIN_PROSTATE_DICE="${HIPMRI_PREFLIGHT_MIN_PROSTATE_DICE:-0.70}"' in script
    assert 'CONFIG_CANDIDATE="${HIPMRI_CONFIG_CANDIDATE:-preflight-a}"' in script
    assert 'python src/experiment_configs.py --candidate "$CONFIG_CANDIDATE" --shell' in script
    assert 'HIPMRI_PROSTATE_CE_WEIGHT="$CONFIG_PROSTATE_CE_WEIGHT"' in script
    assert 'HIPMRI_PROSTATE_DICE_WEIGHT="$CONFIG_PROSTATE_DICE_WEIGHT"' in script
    assert 'HIPMRI_FOREGROUND_PATCH_PROBABILITY="$CONFIG_FOREGROUND_PATCH_PROBABILITY"' in script
    assert "--preflight-patient-id" in script
    assert "--preflight-updates" in script
    assert "--preflight-eval-every" in script
    assert "--train-patch-shape $CONFIG_TRAIN_PATCH_SHAPE" in script
    assert "--foreground-patch-probability \"$HIPMRI_FOREGROUND_PATCH_PROBABILITY\"" in script
    assert "--prostate-ce-weight \"$HIPMRI_PROSTATE_CE_WEIGHT\"" in script
    assert "--prostate-dice-weight \"$HIPMRI_PROSTATE_DICE_WEIGHT\"" in script
    assert "final_train_case_prostate_dice" in script
    assert "Preflight failed: prostate Dice must be at least" in script
    assert "predicted_prostate_voxels" in script
    assert "true_positive_voxels" in script
    assert "python predict.py" not in script


def test_improved_validation_sweep_uses_one_declared_candidate_and_never_reads_test_predictions():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_hipmri_improved_validation_sweep.slurm").read_text(encoding="utf-8")
    candidates = (root / "configs" / "improved_validation_sweep.tsv").read_text(encoding="utf-8").splitlines()

    assert "#SBATCH --array=" not in script
    assert "HIPMRI_SWEEP_CANDIDATE_ID" in script
    assert "HIPMRI_SWEEP_RUN_ID" in script
    assert "python src/audit.py" in script
    assert "python src/train.py" in script
    assert "--selection-metric prostate_dice" in script
    assert "predict.py" not in script
    assert "test_metrics.json" not in script
    assert candidates[0].split("\t") == ["candidate_id", "slug", "learning_rate", "base_channels", "prostate_ce_weight", "prostate_dice_weight", "foreground_patch_probability", "epochs", "selection_metric"]
    assert len(candidates) == 7
    assert candidates[1].split("\t")[1] == "control"


def test_baseline_validation_control_is_matched_and_never_reads_test_predictions():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_hipmri_baseline_validation_control.slurm").read_text(encoding="utf-8")

    assert "#SBATCH --job-name=hipmri-baseline-validation" in script
    assert 'RUN_DIR="$PROJECT_DIR/artifacts/rangpur_baseline_validation_control_context_v2_retry_20261024/${JOB_ID}_baseline"' in script
    assert '--model baseline' in script
    assert '--epochs 20' in script
    assert '--learning-rate 0.001' in script
    assert '--base-channels 16' in script
    assert '--target-shape 128 128 128' in script
    assert '--train-patch-shape 96 96 96' in script
    assert '--foreground-patch-probability 0.50' in script
    assert '--prostate-ce-weight 4.0' in script
    assert '--prostate-dice-weight 4.0' in script
    assert '--selection-metric prostate_dice' in script
    assert "python src/audit.py" in script
    assert "python src/train.py" in script
    assert "predict.py" not in script
    assert "test_metrics.json" not in script


def test_improved_validation_sweep_accepts_a_crlf_candidate_row_before_audit(tmp_path: Path):
    """A CRLF checkout must not reject a valid candidate before the data audit starts."""
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    (tmp_path / "evidence" / "sources").mkdir(parents=True)
    (tmp_path / "data" / "semantic_MRs").mkdir(parents=True)
    (tmp_path / "data" / "semantic_labels_only").mkdir(parents=True)
    shutil.copyfile(root / "evidence" / "sources" / "csiro-brief-data-description.txt", tmp_path / "evidence" / "sources" / "csiro-brief-data-description.txt")
    (tmp_path / "configs" / "improved_validation_sweep.tsv").write_bytes(
        b"candidate_id\tslug\tlearning_rate\tbase_channels\tprostate_ce_weight\tprostate_dice_weight\tforeground_patch_probability\tepochs\tselection_metric\r\n"
        b"0\tcontrol\t0.001\t16\t4.0\t4.0\t0.50\t20\tprostate_dice\r\n"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "git").write_bytes(b"#!/usr/bin/env bash\n[[ $1 == status ]] && exit 0\nexit 0\n")
    (fake_bin / "python").write_bytes(b"#!/usr/bin/env bash\nexit 86\n")
    for command in ("git", "python"):
        (fake_bin / command).chmod(0o755)
    activate = tmp_path / "home" / "miniconda3" / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_bytes(b"conda() { return 0; }\n")
    def wsl_path(path: Path) -> str:
        resolved = path.resolve()
        return f"/mnt/{resolved.drive[0].lower()}/" + "/".join(resolved.parts[1:])

    subprocess.run(["wsl", "chmod", "+x", wsl_path(fake_bin / "git"), wsl_path(fake_bin / "python")], check=True)
    completed = subprocess.run([
        "wsl", "env",
        f"SLURM_SUBMIT_DIR={wsl_path(tmp_path)}",
        "SLURM_JOB_ID=999999",
        "HIPMRI_SWEEP_CANDIDATE_ID=0",
        "HIPMRI_SWEEP_RUN_ID=crlf-regression",
        f"HIPMRI_DATA_ROOT={wsl_path(tmp_path / 'data')}",
        f"HOME={wsl_path(tmp_path / 'home')}",
        f"PATH={wsl_path(fake_bin)}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "bash", wsl_path(root / "scripts" / "run_hipmri_improved_validation_sweep.slurm"),
    ], capture_output=True)

    stderr = completed.stderr.decode("utf-8", errors="replace")
    assert completed.returncode == 86, stderr
    assert "Sweep row does not match" not in stderr


def test_matched_wide20_formal_script_separates_control_from_one_test_per_formal_arm():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_hipmri_wide20_matched_formal.slurm").read_text(encoding="utf-8")

    assert 'FORMAL_MODE="${HIPMRI_FORMAL_MODE:?set control, baseline_new, improved_wide20, or improved_wide20_recovery}"' in script
    assert 'RUN_DIR="$RUN_ROOT/${JOB_ID}_${FORMAL_MODE}"' in script
    assert 'CONTROL=1' in script
    assert '--selection-metric prostate_dice' in script
    assert '--model "$MODEL_NAME"' in script
    assert '--base-channels "$BASE_CHANNELS"' in script
    assert 'if [[ "$CONTROL" == "0" ]]; then' in script
    assert 'python src/predict.py' in script
    assert 'sha256sum "$TRAINING_DIR/best.pt"' in script
    assert 'sacct -X -n -P -j "$JOB_ID"' in script


def test_matched_wide20_formal_script_can_test_a_verified_frozen_checkpoint_without_training_again():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_hipmri_wide20_matched_formal.slurm").read_text(encoding="utf-8")

    assert "improved_wide20_recovery)" in script
    assert 'RECOVERED_CHECKPOINT="${HIPMRI_RECOVERED_WIDE20_CHECKPOINT:?set the verified wide20 best.pt path}"' in script
    assert 'if [[ "$RECOVERY" == "1" ]]; then' in script
    assert 'cp "$RECOVERED_CHECKPOINT" "$TRAINING_DIR/best.pt"' in script
    assert 'python src/train.py' in script
