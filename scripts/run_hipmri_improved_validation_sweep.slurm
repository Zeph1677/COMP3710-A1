#!/usr/bin/env bash
#SBATCH --job-name=hipmri-improved-sweep
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm-improved-sweep-%j.out
#SBATCH --error=slurm-improved-sweep-%j.err

# Improved-only validation sweep. It intentionally never reads the previously
# inspected held-out patient partition.
set -euo pipefail
export PATH="/opt/slurm/bin:$PATH"

PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit this file from the project directory}"
DATA_ROOT="${HIPMRI_DATA_ROOT:-/home/groups/comp3710/HipMRI_Study_open}"
CANDIDATE_ID="${HIPMRI_SWEEP_CANDIDATE_ID:?set a candidate ID from improved_validation_sweep.tsv}"
SWEEP_RUN_ID="${HIPMRI_SWEEP_RUN_ID:?set a shared non-empty sweep run ID}"
JOB_ID="${SLURM_JOB_ID:?this script must run under Slurm}"
SWEEP_TABLE="$PROJECT_DIR/configs/improved_validation_sweep.tsv"
PROSTATE_SOURCE="CSIRO, Labelled weekly MR images of the male pelvis, v2, DOI: 10.25919/45T8-P065, BriefDataDescription.txt, Labels block: 5=Prostate."
EXPECTED_SOURCE_NORMALIZED_SHA256="89C5EF8336573B6C359D0828E2FFCF6819149490FEE20071F7436ED411DD89AF"

cd "$PROJECT_DIR"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to run from a dirty checkout; commit or stash changes first." >&2
    exit 2
fi
if [[ ! -f "$SWEEP_TABLE" ]]; then
    echo "Missing sweep table: $SWEEP_TABLE" >&2
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

line_number=$((CANDIDATE_ID + 2))
candidate_line="$(sed -n "${line_number}p" "$SWEEP_TABLE")"
if [[ -z "$candidate_line" ]]; then
    echo "No sweep candidate for ID $CANDIDATE_ID." >&2
    exit 2
fi
IFS=$'\t' read -r CANDIDATE_ID SLUG LEARNING_RATE BASE_CHANNELS CE_WEIGHT DICE_WEIGHT FOREGROUND_PROBABILITY EPOCHS SELECTION_METRIC <<< "$candidate_line"
if [[ "$CANDIDATE_ID" != "${HIPMRI_SWEEP_CANDIDATE_ID}" || "$SELECTION_METRIC" != "prostate_dice" ]]; then
    echo "Sweep row does not match the requested candidate or selection metric." >&2
    exit 2
fi

RUN_ROOT="$PROJECT_DIR/artifacts/rangpur_improved_validation_sweep_${SWEEP_RUN_ID}"
RUN_DIR="$RUN_ROOT/candidate_$(printf '%02d' "$CANDIDATE_ID")_${SLUG}"
TRAINING_DIR="$RUN_DIR/training_improved"
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing candidate output: $RUN_DIR" >&2
    exit 2
fi
mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/stdout.log") 2> >(tee "$RUN_DIR/stderr.log" >&2)
trap 'sacct -P -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES > "$RUN_DIR/sacct.txt" || true' EXIT

source "$HOME/miniconda3/bin/activate"
conda activate torch

git rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git status --porcelain > "$RUN_DIR/git_status.txt"
cp "$SWEEP_TABLE" "$RUN_DIR/improved_validation_sweep.tsv"
printf '%s\n' "$PROSTATE_SOURCE" > "$RUN_DIR/prostate_label_source.txt"
printf '%s\n' "prostate_label=5" >> "$RUN_DIR/prostate_label_source.txt"
python --version > "$RUN_DIR/python_version.txt"
python -c "import json, platform, torch; print(json.dumps({'platform': platform.platform(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))" > "$RUN_DIR/environment.json"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RUN_DIR/gpu_before.csv"

python src/audit.py "$DATA_ROOT" --output "$RUN_DIR/re_audited_manifest.json" --seed 42
cmp --silent evidence/dataset_manifest.json "$RUN_DIR/re_audited_manifest.json" || {
    echo "Shared data audit does not match the committed manifest; refusing to train." >&2
    exit 2
}

cat > "$RUN_DIR/commands.txt" <<EOF
python src/train.py --manifest evidence/dataset_manifest.json --data-root $DATA_ROOT --output-dir $TRAINING_DIR --model improved --epochs $EPOCHS --batch-size 1 --learning-rate $LEARNING_RATE --base-channels $BASE_CHANNELS --target-shape 128 128 128 --train-patch-shape 96 96 96 --foreground-patch-probability $FOREGROUND_PROBABILITY --seed 42 --device cuda --prostate-label 5 --prostate-label-source "$PROSTATE_SOURCE" --prostate-ce-weight $CE_WEIGHT --prostate-dice-weight $DICE_WEIGHT --selection-metric prostate_dice
EOF

time python src/train.py --manifest evidence/dataset_manifest.json --data-root "$DATA_ROOT" --output-dir "$TRAINING_DIR" --model improved --epochs "$EPOCHS" --batch-size 1 --learning-rate "$LEARNING_RATE" --base-channels "$BASE_CHANNELS" --target-shape 128 128 128 --train-patch-shape 96 96 96 --foreground-patch-probability "$FOREGROUND_PROBABILITY" --seed 42 --device cuda --prostate-label 5 --prostate-label-source "$PROSTATE_SOURCE" --prostate-ce-weight "$CE_WEIGHT" --prostate-dice-weight "$DICE_WEIGHT" --selection-metric "$SELECTION_METRIC"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RUN_DIR/gpu_after.csv"
[[ -f "slurm-improved-sweep-${JOB_ID}.out" ]] && cp "slurm-improved-sweep-${JOB_ID}.out" "$RUN_DIR/slurm-${JOB_ID}.out" || true
[[ -f "slurm-improved-sweep-${JOB_ID}.err" ]] && cp "slurm-improved-sweep-${JOB_ID}.err" "$RUN_DIR/slurm-${JOB_ID}.err" || true
