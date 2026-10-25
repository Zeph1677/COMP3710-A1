#!/usr/bin/env bash
#SBATCH --job-name=hipmri-P123
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=slurm-candidates-%j.out
#SBATCH --error=slurm-candidates-%j.err
set -euo pipefail
export PATH="/opt/slurm/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:?}"
candidate="${HIPMRI_CANDIDATE:?}"
case "$candidate" in P1|P2|P3) ;; *) echo "Unknown candidate" >&2; exit 2 ;; esac
run_dir="$PWD/artifacts/validation_candidates_${SLURM_JOB_ID:?}/$candidate"
[[ ! -e "$run_dir" ]] || { echo 'Refusing existing run directory' >&2; exit 2; }
mkdir -p "$run_dir"
exec > >(tee "$run_dir/stdout.log") 2> >(tee "$run_dir/stderr.log" >&2)
finish() {
    result=$?
    trap - EXIT
    printf '{"exit_code":%s,"job_id":"%s","candidate":"%s"}\n' "$result" "$SLURM_JOB_ID" "$candidate" > "$run_dir/process_exit.json"
    sacct -X -n -P -j "$SLURM_JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES > "$run_dir/sacct_at_exit.txt" || true
    exit "$result"
}
trap finish EXIT
source "$HOME/miniconda3/bin/activate"
conda activate torch
python --version > "$run_dir/python_version.txt"
python -m pip freeze > "$run_dir/pip_freeze.txt"
nvidia-smi > "$run_dir/gpu.txt"
cp SOURCE_SNAPSHOT.json "$run_dir/source_snapshot.json"
python -c 'import json,sys,torch; print(json.dumps({"python":sys.version,"torch":str(torch.__version__),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"amp":False},indent=2))' > "$run_dir/environment.json"
sacct -X -n -P -j 692354 --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES > "$run_dir/reused_692354_sacct_final.txt"
data_root="${HIPMRI_DATA_ROOT:-/home/groups/comp3710/HipMRI_Study_open}"
python src/audit.py "$data_root" --output "$run_dir/re_audited_manifest.json" --seed 42
cmp evidence/dataset_manifest.json "$run_dir/re_audited_manifest.json"
python src/validation_candidates.py --candidate "$candidate" --output-root "$run_dir/result" --data-root "$data_root" --existing-accounting "$run_dir/reused_692354_sacct_final.txt" --existing-root /home/Student/s4910970/hipmri_context_v2_ablation_20261024/artifacts/context_v2_ablation_692354/runs
