#!/usr/bin/env bash
#SBATCH --job-name=hipmri-preflight
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=slurm-preflight-%j.out
#SBATCH --error=slurm-preflight-%j.err

# Submit this file from the repository project directory. SLURM_SUBMIT_DIR,
# HOME and SLURM_JOB_ID are evaluated by Rangpur, not by Windows PowerShell.
set -euo pipefail
export PATH="/opt/slurm/bin:$PATH"

PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit this file with sbatch from the project directory}"
DATA_ROOT="${HIPMRI_DATA_ROOT:-/home/groups/comp3710/HipMRI_Study_open}"
JOB_ID="${SLURM_JOB_ID:?this script must run under Slurm}"
RUN_DIR="$PROJECT_DIR/artifacts/${JOB_ID}_preflight"
TRAINING_DIR="$RUN_DIR/training"
HIPMRI_PREFLIGHT_PATIENT_ID="${HIPMRI_PREFLIGHT_PATIENT_ID:?set a manifest train patient ID}"
HIPMRI_PREFLIGHT_UPDATES="${HIPMRI_PREFLIGHT_UPDATES:-200}"
HIPMRI_PREFLIGHT_EVAL_EVERY="${HIPMRI_PREFLIGHT_EVAL_EVERY:-20}"
HIPMRI_PREFLIGHT_MIN_PROSTATE_DICE="${HIPMRI_PREFLIGHT_MIN_PROSTATE_DICE:-0.70}"
CONFIG_CANDIDATE="${HIPMRI_CONFIG_CANDIDATE:-preflight-a}"
PREFLIGHT_MODEL="baseline"
PROSTATE_SOURCE="CSIRO, Labelled weekly MR images of the male pelvis, v2, DOI: 10.25919/45T8-P065, BriefDataDescription.txt, Labels block: 5=Prostate."
EXPECTED_SOURCE_NORMALIZED_SHA256="89C5EF8336573B6C359D0828E2FFCF6819149490FEE20071F7436ED411DD89AF"

cd "$PROJECT_DIR"
eval "$(python src/experiment_configs.py --candidate "$CONFIG_CANDIDATE" --shell)"
HIPMRI_PROSTATE_CE_WEIGHT="$CONFIG_PROSTATE_CE_WEIGHT"
HIPMRI_PROSTATE_DICE_WEIGHT="$CONFIG_PROSTATE_DICE_WEIGHT"
HIPMRI_FOREGROUND_PATCH_PROBABILITY="$CONFIG_FOREGROUND_PATCH_PROBABILITY"
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

python - "$HIPMRI_PREFLIGHT_UPDATES" "$HIPMRI_PREFLIGHT_EVAL_EVERY" "$HIPMRI_PREFLIGHT_MIN_PROSTATE_DICE" "$HIPMRI_PROSTATE_CE_WEIGHT" "$HIPMRI_PROSTATE_DICE_WEIGHT" "$HIPMRI_FOREGROUND_PATCH_PROBABILITY" <<'PY'
import sys

updates, eval_every, minimum_dice, ce_weight, dice_weight, foreground_probability = sys.argv[1:]
if not updates.isdigit() or int(updates) <= 0:
    raise SystemExit("HIPMRI_PREFLIGHT_UPDATES must be a positive integer.")
if not eval_every.isdigit() or int(eval_every) <= 0:
    raise SystemExit("HIPMRI_PREFLIGHT_EVAL_EVERY must be a positive integer.")
if float(minimum_dice) < 0.0 or float(minimum_dice) > 1.0:
    raise SystemExit("HIPMRI_PREFLIGHT_MIN_PROSTATE_DICE must be in [0, 1].")
if float(ce_weight) <= 0.0 or float(dice_weight) <= 0.0:
    raise SystemExit("Prostate loss weights must be positive.")
if float(foreground_probability) < 0.0 or float(foreground_probability) > 1.0:
    raise SystemExit("HIPMRI_FOREGROUND_PATCH_PROBABILITY must be in [0, 1].")
PY

mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/stdout.log") 2> >(tee "$RUN_DIR/stderr.log" >&2)
trap 'sacct -P -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES > "$RUN_DIR/sacct.txt" || true' EXIT

source "$HOME/miniconda3/bin/activate"
conda activate torch
git rev-parse HEAD > "$RUN_DIR/git_sha.txt"
python --version > "$RUN_DIR/python_version.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RUN_DIR/gpu_before.csv"

python src/audit.py "$DATA_ROOT" --output "$RUN_DIR/re_audited_manifest.json" --seed 42
cmp --silent evidence/dataset_manifest.json "$RUN_DIR/re_audited_manifest.json" || {
    echo "Shared data audit does not match the committed manifest; refusing to train." >&2
    exit 2
}

cat > "$RUN_DIR/commands.txt" <<EOF
python src/train.py --manifest evidence/dataset_manifest.json --data-root $DATA_ROOT --output-dir $TRAINING_DIR --model $PREFLIGHT_MODEL --epochs 1 --preflight-updates $HIPMRI_PREFLIGHT_UPDATES --preflight-eval-every $HIPMRI_PREFLIGHT_EVAL_EVERY --batch-size 1 --learning-rate 0.001 --base-channels 16 --target-shape $CONFIG_TARGET_SHAPE --train-patch-shape $CONFIG_TRAIN_PATCH_SHAPE --foreground-patch-probability $HIPMRI_FOREGROUND_PATCH_PROBABILITY --seed $CONFIG_SEED --device cuda --prostate-label 5 --prostate-label-source "$PROSTATE_SOURCE" --prostate-ce-weight $HIPMRI_PROSTATE_CE_WEIGHT --prostate-dice-weight $HIPMRI_PROSTATE_DICE_WEIGHT --preflight-patient-id $HIPMRI_PREFLIGHT_PATIENT_ID
minimum_preflight_prostate_dice=$HIPMRI_PREFLIGHT_MIN_PROSTATE_DICE
EOF

python src/train.py --manifest evidence/dataset_manifest.json --data-root "$DATA_ROOT" --output-dir "$TRAINING_DIR" --model "$PREFLIGHT_MODEL" --epochs 1 --preflight-updates "$HIPMRI_PREFLIGHT_UPDATES" --preflight-eval-every "$HIPMRI_PREFLIGHT_EVAL_EVERY" --batch-size 1 --learning-rate 0.001 --base-channels 16 --target-shape $CONFIG_TARGET_SHAPE --train-patch-shape $CONFIG_TRAIN_PATCH_SHAPE --foreground-patch-probability "$HIPMRI_FOREGROUND_PATCH_PROBABILITY" --seed "$CONFIG_SEED" --device cuda --prostate-label 5 --prostate-label-source "$PROSTATE_SOURCE" --prostate-ce-weight "$HIPMRI_PROSTATE_CE_WEIGHT" --prostate-dice-weight "$HIPMRI_PROSTATE_DICE_WEIGHT" --preflight-patient-id "$HIPMRI_PREFLIGHT_PATIENT_ID"

python - "$TRAINING_DIR/preflight_summary.json" "$HIPMRI_PREFLIGHT_MIN_PROSTATE_DICE" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
minimum_dice = float(sys.argv[2])
dice = float(summary["final_train_case_prostate_dice"])
if int(summary["true_positive_voxels"]) <= 0 or int(summary["predicted_prostate_voxels"]) <= 0:
    raise SystemExit("Preflight failed: prostate prediction must contain at least one true-positive voxel.")
if dice < minimum_dice:
    raise SystemExit(f"Preflight failed: prostate Dice must be at least {minimum_dice:.4f}; observed {dice:.4f}.")
PY

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RUN_DIR/gpu_after.csv"
