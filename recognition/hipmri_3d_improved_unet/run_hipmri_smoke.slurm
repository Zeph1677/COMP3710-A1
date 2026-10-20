#!/usr/bin/env bash
#SBATCH --job-name=hipmri-baseline-smoke
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Submit from this project directory. The Slurm stdout/stderr files at the
# project root are ignored; the immutable run record is copied into artifacts.
# The live comp3710 partition advertises only 1 MB/node to Slurm. Any --mem
# request is rejected before scheduling, so memory is intentionally omitted.
set -euo pipefail
export PATH="/opt/slurm/bin:$PATH"

PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit this file with sbatch from the project directory}"
DATA_ROOT="${HIPMRI_DATA_ROOT:-/home/groups/comp3710/HipMRI_Study_open}"
JOB_ID="${SLURM_JOB_ID:?this script must run under Slurm}"
RUN_DIR="$PROJECT_DIR/artifacts/rangpur_smoke_${JOB_ID}"
MODEL_NAME="${HIPMRI_MODEL:-baseline}"
PROSTATE_SOURCE="CSIRO, Labelled weekly MR images of the male pelvis, v2, DOI: 10.25919/45T8-P065, BriefDataDescription.txt, Labels block: 5=Prostate."
# Git may normalize CRLF to LF on Linux. This checksum validates the same text
# after LF normalization; SOURCES.md retains the raw downloaded-file checksum.
EXPECTED_SOURCE_NORMALIZED_SHA256="89C5EF8336573B6C359D0828E2FFCF6819149490FEE20071F7436ED411DD89AF"

cd "$PROJECT_DIR"
if [[ "$MODEL_NAME" != "baseline" && "$MODEL_NAME" != "improved" ]]; then
    echo "HIPMRI_MODEL must be baseline or improved, found: $MODEL_NAME" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to run from a dirty checkout; commit or stash changes first." >&2
    exit 2
fi
if [[ ! -d "$DATA_ROOT/semantic_MRs" || ! -d "$DATA_ROOT/semantic_labels_only" ]]; then
    echo "HipMRI data root is incomplete: $DATA_ROOT" >&2
    exit 2
fi
if [[ "$(tr -d '\r' < evidence/sources/csiro-brief-data-description.txt | sha256sum | awk '{print toupper($1)}')" != "$EXPECTED_SOURCE_NORMALIZED_SHA256" ]]; then
    echo "The frozen CSIRO label-mapping evidence does not match its normalized SHA-256." >&2
    exit 2
fi

mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/stdout.log") 2> >(tee "$RUN_DIR/stderr.log" >&2)

source "$HOME/miniconda3/bin/activate"
conda activate torch

git rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git status --porcelain > "$RUN_DIR/git_status.txt"
printf '%s\n' "$PROSTATE_SOURCE" > "$RUN_DIR/prostate_label_source.txt"
printf '%s\n' "prostate_label=5" >> "$RUN_DIR/prostate_label_source.txt"
python --version > "$RUN_DIR/python_version.txt"
python -c "import json, platform, torch; print(json.dumps({'platform': platform.platform(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))" > "$RUN_DIR/environment.json"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RUN_DIR/gpu_before.csv"

python audit.py "$DATA_ROOT" --output "$RUN_DIR/re_audited_manifest.json" --seed 42
cmp --silent evidence/dataset_manifest.json "$RUN_DIR/re_audited_manifest.json" || {
    echo "Shared data audit does not match the committed manifest; refusing to train." >&2
    exit 2
}

cat > "$RUN_DIR/commands.txt" <<EOF
python train.py --manifest evidence/dataset_manifest.json --data-root $DATA_ROOT --output-dir $RUN_DIR/training_$MODEL_NAME --model $MODEL_NAME --epochs 1 --batch-size 1 --learning-rate 0.001 --base-channels 16 --target-shape 128 128 128 --seed 42 --device cuda --prostate-label 5 --prostate-label-source "$PROSTATE_SOURCE"
python predict.py --checkpoint $RUN_DIR/training_$MODEL_NAME/best.pt --manifest evidence/dataset_manifest.json --data-root $DATA_ROOT --output-dir $RUN_DIR/test_$MODEL_NAME --patch-size 128 128 128 --stride 64 64 64 --device cuda
EOF

time python train.py --manifest evidence/dataset_manifest.json --data-root "$DATA_ROOT" --output-dir "$RUN_DIR/training_$MODEL_NAME" --model "$MODEL_NAME" --epochs 1 --batch-size 1 --learning-rate 0.001 --base-channels 16 --target-shape 128 128 128 --seed 42 --device cuda --prostate-label 5 --prostate-label-source "$PROSTATE_SOURCE"
time python predict.py --checkpoint "$RUN_DIR/training_$MODEL_NAME/best.pt" --manifest evidence/dataset_manifest.json --data-root "$DATA_ROOT" --output-dir "$RUN_DIR/test_$MODEL_NAME" --patch-size 128 128 128 --stride 64 64 64 --device cuda

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RUN_DIR/gpu_after.csv"
cp "slurm-${JOB_ID}.out" "$RUN_DIR/slurm-${JOB_ID}.out" || true
cp "slurm-${JOB_ID}.err" "$RUN_DIR/slurm-${JOB_ID}.err" || true
