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
    assert "python train.py" not in script
    assert "python predict.py" not in script
    assert 'MODEL_NAME="${HIPMRI_MODEL:-baseline}"' in script
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
    assert 'RUN_DIR="$PROJECT_DIR/artifacts/rangpur_preflight_${JOB_ID}"' in script
    assert 'TRAINING_DIR="$RUN_DIR/training"' in script
    assert '--output-dir "$TRAINING_DIR"' in script
    assert "python src/train.py" in script
    assert "python train.py" not in script
    assert 'source "$HOME/miniconda3/bin/activate"' in script
    assert 'HIPMRI_PREFLIGHT_PATIENT_ID="${HIPMRI_PREFLIGHT_PATIENT_ID:?set a manifest train patient ID}"' in script
    assert "--preflight-patient-id" in script
    assert "python predict.py" not in script
