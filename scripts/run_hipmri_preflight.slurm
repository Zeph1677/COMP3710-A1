#!/usr/bin/env bash
#SBATCH --job-name=hipmri-preflight
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm-preflight-%j.out
#SBATCH --error=slurm-preflight-%j.err

# Submit this file from the repository project directory. SLURM_SUBMIT_DIR,
# HOME and SLURM_JOB_ID are evaluated by Rangpur, not by Windows PowerShell.
set -euo pipefail
export PATH="/opt/slurm/bin:$PATH"

PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit this file with sbatch from the project directory}"
DATA_ROOT="${HIPMRI_DATA_ROOT:-/home/groups/comp3710/HipMRI_Study_open}"
JOB_ID="${SLURM_JOB_ID:?this script must run under Slurm}"
RUN_DIR="$PROJECT_DIR/artifacts/rangpur_preflight_${JOB_ID}"
TRAINING_DIR="$RUN_DIR/training"
HIPMRI_PREFLIGHT_PATIENT_ID="${HIPMRI_PREFLIGHT_PATIENT_ID:?set a manifest train patient ID}"
PROSTATE_SOURCE="CSIRO, Labelled weekly MR images of the male pelvis, v2, DOI: 10.25919/45T8-P065, BriefDataDescription.txt, Labels block: 5=Prostate."

cd "$PROJECT_DIR"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to run from a dirty checkout; commit or stash changes first." >&2
    exit 2
fi
if [[ ! -d "$DATA_ROOT/semantic_MRs" || ! -d "$DATA_ROOT/semantic_labels_only" ]]; then
    echo "HipMRI data root is incomplete: $DATA_ROOT" >&2
    exit 2
fi

mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/stdout.log") 2> >(tee "$RUN_DIR/stderr.log" >&2)
trap 'sacct -P -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES > "$RUN_DIR/sacct.txt" || true' EXIT

source "$HOME/miniconda3/bin/activate"
conda activate torch
git rev-parse HEAD > "$RUN_DIR/git_sha.txt"
python --version > "$RUN_DIR/python_version.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RUN_DIR/gpu_before.csv"

python src/train.py --manifest evidence/dataset_manifest.json --data-root "$DATA_ROOT" --output-dir "$TRAINING_DIR" --model baseline --epochs "${HIPMRI_PREFLIGHT_EPOCHS:-10}" --batch-size 1 --learning-rate 0.001 --base-channels 16 --target-shape 128 128 128 --seed 42 --device cuda --prostate-label 5 --prostate-label-source "$PROSTATE_SOURCE" --preflight-patient-id "$HIPMRI_PREFLIGHT_PATIENT_ID"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RUN_DIR/gpu_after.csv"
