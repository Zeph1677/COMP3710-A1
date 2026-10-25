#!/usr/bin/env bash
#SBATCH --job-name=hipmri-validation-completion
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=slurm-validation-completion-%j.out
#SBATCH --error=slurm-validation-completion-%j.err
set -euo pipefail
export PATH="/opt/slurm/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:?}"
DATA_ROOT="${HIPMRI_DATA_ROOT:-/home/groups/comp3710/HipMRI_Study_open}"
EXISTING_ROOT="${HIPMRI_EXISTING_RUN_ROOT:-/home/Student/s4910970/hipmri_context_v2_ablation_20261024/artifacts/context_v2_ablation_692354/runs}"
RUN_DIR="$PWD/artifacts/validation_completion_${SLURM_JOB_ID:?}"
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing run: $RUN_DIR" >&2
    exit 2
fi
mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/stdout.log") 2> >(tee "$RUN_DIR/stderr.log" >&2)
finish() {
    result=$?
    trap - EXIT
    printf '{"exit_code":%s,"job_id":"%s"}\n' "$result" "$SLURM_JOB_ID" > "$RUN_DIR/process_exit.json"
    sacct -X -n -P -j "$SLURM_JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES > "$RUN_DIR/sacct_at_exit.txt" || true
    exit "$result"
}
trap finish EXIT
source "$HOME/miniconda3/bin/activate"
conda activate torch
python --version > "$RUN_DIR/python_version.txt"
python -m pip freeze > "$RUN_DIR/pip_freeze.txt"
nvidia-smi > "$RUN_DIR/gpu.txt"
cp SOURCE_SNAPSHOT.json "$RUN_DIR/source_snapshot.json"
cp configs/validation_completion_protocol.json "$RUN_DIR/registered_protocol.json"
python -c 'import json,platform,sys,torch; print(json.dumps({"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"torch":str(torch.__version__),"cuda_build":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"gpu":torch.cuda.get_device_name(0),"precision":"float32","amp":False},indent=2))' > "$RUN_DIR/environment.json"
sacct -X -n -P -j 692354 --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES > "$RUN_DIR/reused_692354_sacct_final.txt"
python src/audit.py "$DATA_ROOT" --output "$RUN_DIR/re_audited_manifest.json" --seed 42
cmp evidence/dataset_manifest.json "$RUN_DIR/re_audited_manifest.json"
python src/validation_completion.py --existing-root "$EXISTING_ROOT" --existing-accounting "$RUN_DIR/reused_692354_sacct_final.txt" --manifest evidence/dataset_manifest.json --data-root "$DATA_ROOT" --protocol configs/validation_completion_protocol.json --output-root "$RUN_DIR/validation"
