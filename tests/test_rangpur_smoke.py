from pathlib import Path


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
