#!/usr/bin/env bash
#SBATCH --job-name=hipmri-wide20-formal
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm-wide20-formal-%j.out
#SBATCH --error=slurm-wide20-formal-%j.err

# This script owns the new matched protocol only. It never overwrites the
# historical formal or validation-sweep directories.
set -euo pipefail
export PATH="/opt/slurm/bin:$PATH"

PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit from the project directory}"
DATA_ROOT="${HIPMRI_DATA_ROOT:-/home/groups/comp3710/HipMRI_Study_open}"
FORMAL_MODE="${HIPMRI_FORMAL_MODE:?set control, baseline_new, improved_wide20, or improved_wide20_recovery}"
JOB_ID="${SLURM_JOB_ID:?this script must run under Slurm}"
RUN_ROOT="$PROJECT_DIR/artifacts/rangpur_formal_wide20_matched_20261022"
RUN_DIR="$RUN_ROOT/${JOB_ID}_${FORMAL_MODE}"
TRAINING_CODE_SHA="2e2529857a586a009f0f4cad2e260dc810531ac0"
PROSTATE_SOURCE="CSIRO, Labelled weekly MR images of the male pelvis, v2, DOI: 10.25919/45T8-P065, BriefDataDescription.txt, Labels block: 5=Prostate."
EXPECTED_SOURCE_NORMALIZED_SHA256="89C5EF8336573B6C359D0828E2FFCF6819149490FEE20071F7436ED411DD89AF"

MODEL_NAME=""
BASE_CHANNELS=""
CONTROL=0
RECOVERY=0
RECOVERED_CHECKPOINT=""
case "$FORMAL_MODE" in
    control)
        MODEL_NAME="improved"
        BASE_CHANNELS=16
        CONTROL=1
        ;;
    baseline_new)
        MODEL_NAME="baseline"
        BASE_CHANNELS=16
        ;;
    improved_wide20)
        MODEL_NAME="improved"
        BASE_CHANNELS=20
        ;;
    improved_wide20_recovery)
        MODEL_NAME="improved"
        BASE_CHANNELS=20
        RECOVERY=1
        RECOVERED_CHECKPOINT="${HIPMRI_RECOVERED_WIDE20_CHECKPOINT:?set the verified wide20 best.pt path}"
        ;;
    *)
        echo "HIPMRI_FORMAL_MODE must be control, baseline_new, improved_wide20, or improved_wide20_recovery." >&2
        exit 2
        ;;
esac

cd "$PROJECT_DIR"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to run from a dirty checkout." >&2
    exit 2
fi
if ! git diff --quiet "$TRAINING_CODE_SHA" -- src/train.py src/predict.py src/modules.py evidence/dataset_manifest.json; then
    echo "Model, prediction, module, or manifest code differs from frozen training SHA $TRAINING_CODE_SHA." >&2
    exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing formal output: $RUN_DIR" >&2
    exit 2
fi
if [[ "$RECOVERY" == "1" && ! -f "$RECOVERED_CHECKPOINT" ]]; then
    echo "Verified recovered wide20 checkpoint is missing: $RECOVERED_CHECKPOINT" >&2
    exit 2
fi
if [[ ! -d "$DATA_ROOT/semantic_MRs" || ! -d "$DATA_ROOT/semantic_labels_only" ]]; then
    echo "HipMRI data root is incomplete: $DATA_ROOT" >&2
    exit 2
fi
if [[ "$(tr -d '\r' < evidence/sources/csiro-brief-data-description.txt | sha256sum | awk '{print toupper($1)}')" != "$EXPECTED_SOURCE_NORMALIZED_SHA256" ]]; then
    echo "Frozen CSIRO label-mapping evidence hash does not match." >&2
    exit 2
fi

mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/stdout.log") 2> >(tee "$RUN_DIR/stderr.log" >&2)
trap 'sacct -X -n -P -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES > "$RUN_DIR/sacct_exit_snapshot.txt" || true' EXIT

source "$HOME/miniconda3/bin/activate"
conda activate torch

git rev-parse HEAD > "$RUN_DIR/orchestration_git_sha.txt"
printf '%s\n' "$TRAINING_CODE_SHA" > "$RUN_DIR/training_code_git_sha.txt"
git status --porcelain > "$RUN_DIR/git_status.txt"
printf '%s\n' "$PROSTATE_SOURCE" > "$RUN_DIR/prostate_label_source.txt"
printf '%s\n' "prostate_label=5" >> "$RUN_DIR/prostate_label_source.txt"
python --version > "$RUN_DIR/python_version.txt"
python -c "import json, platform, torch; print(json.dumps({'platform': platform.platform(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))" > "$RUN_DIR/environment.json"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RUN_DIR/gpu_before.csv"

python src/audit.py "$DATA_ROOT" --output "$RUN_DIR/re_audited_manifest.json" --seed 42
cmp --silent evidence/dataset_manifest.json "$RUN_DIR/re_audited_manifest.json" || {
    echo "Regenerated manifest differs from the committed manifest." >&2
    exit 2
}

TRAINING_DIR="$RUN_DIR/training_${MODEL_NAME}"
cat > "$RUN_DIR/commands.txt" <<EOF
python src/train.py --manifest evidence/dataset_manifest.json --data-root $DATA_ROOT --output-dir $TRAINING_DIR --model $MODEL_NAME --epochs 20 --batch-size 1 --learning-rate 0.001 --base-channels $BASE_CHANNELS --target-shape 128 128 128 --train-patch-shape 96 96 96 --foreground-patch-probability 0.50 --seed 42 --device cuda --prostate-label 5 --prostate-label-source "$PROSTATE_SOURCE" --prostate-ce-weight 4.0 --prostate-dice-weight 4.0 --selection-metric prostate_dice
EOF

if [[ "$RECOVERY" == "1" ]]; then
    recovered_training_dir="$(dirname "$RECOVERED_CHECKPOINT")"
    for required in best.pt experiment_config.json history.json training_summary.json; do
        if [[ ! -f "$recovered_training_dir/$required" ]]; then
            echo "Recovered wide20 training evidence is incomplete: $recovered_training_dir/$required" >&2
            exit 2
        fi
    done
    mkdir -p "$TRAINING_DIR"
    cp "$RECOVERED_CHECKPOINT" "$TRAINING_DIR/best.pt"
    cp "$recovered_training_dir/experiment_config.json" "$TRAINING_DIR/experiment_config.json"
    cp "$recovered_training_dir/history.json" "$TRAINING_DIR/history.json"
    cp "$recovered_training_dir/training_summary.json" "$TRAINING_DIR/training_summary.json"
else
    time python src/train.py --manifest evidence/dataset_manifest.json --data-root "$DATA_ROOT" --output-dir "$TRAINING_DIR" --model "$MODEL_NAME" --epochs 20 --batch-size 1 --learning-rate 0.001 --base-channels "$BASE_CHANNELS" --target-shape 128 128 128 --train-patch-shape 96 96 96 --foreground-patch-probability 0.50 --seed 42 --device cuda --prostate-label 5 --prostate-label-source "$PROSTATE_SOURCE" --prostate-ce-weight 4.0 --prostate-dice-weight 4.0 --selection-metric prostate_dice
fi
sha256sum "$TRAINING_DIR/best.pt" > "$RUN_DIR/checkpoint_sha256.txt"

if [[ "$CONTROL" == "0" ]]; then
    TESTING_DIR="$RUN_DIR/test_${MODEL_NAME}"
    cat >> "$RUN_DIR/commands.txt" <<EOF
python src/predict.py --checkpoint $TRAINING_DIR/best.pt --manifest evidence/dataset_manifest.json --data-root $DATA_ROOT --output-dir $TESTING_DIR --patch-size 128 128 128 --stride 64 64 64 --device cuda
EOF
    time python src/predict.py --checkpoint "$TRAINING_DIR/best.pt" --manifest evidence/dataset_manifest.json --data-root "$DATA_ROOT" --output-dir "$TESTING_DIR" --patch-size 128 128 128 --stride 64 64 64 --device cuda
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RUN_DIR/gpu_after.csv"
[[ -f "slurm-wide20-formal-${JOB_ID}.out" ]] && cp "slurm-wide20-formal-${JOB_ID}.out" "$RUN_DIR/slurm-${JOB_ID}.out" || true
[[ -f "slurm-wide20-formal-${JOB_ID}.err" ]] && cp "slurm-wide20-formal-${JOB_ID}.err" "$RUN_DIR/slurm-${JOB_ID}.err" || true
