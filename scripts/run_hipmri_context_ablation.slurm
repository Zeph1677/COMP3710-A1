#!/usr/bin/env bash
#SBATCH --job-name=hipmri-context-ablation
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=slurm-context-ablation-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source "$HOME/miniconda3/bin/activate"
conda activate torch
RUN_DIR="artifacts/context_v2_ablation_${SLURM_JOB_ID:?}"
mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git diff --binary > "$RUN_DIR/source_changes.patch"
python --version > "$RUN_DIR/python_version.txt"
nvidia-smi > "$RUN_DIR/gpu.txt"
python src/audit.py "${HIPMRI_DATA_ROOT:-/home/groups/comp3710/HipMRI_Study_open}" --output "$RUN_DIR/re_audited_manifest.json" --seed 42
cmp evidence/dataset_manifest.json "$RUN_DIR/re_audited_manifest.json"
python src/run_validation_ablation.py --data-root "${HIPMRI_DATA_ROOT:-/home/groups/comp3710/HipMRI_Study_open}" --output-root "$RUN_DIR/runs" --device cuda
