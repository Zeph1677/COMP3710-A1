from pathlib import Path


def test_rangpur_smoke_script_records_declared_mapping_and_reproducibility_evidence():
    script = (Path(__file__).resolve().parents[1] / "run_hipmri_smoke.slurm").read_text(encoding="utf-8")

    assert "#SBATCH --account=comp3710" in script
    assert "#SBATCH --partition=comp3710" in script
    assert 'export PATH="/opt/slurm/bin:$PATH"' in script
    assert "/home/groups/comp3710/HipMRI_Study_open" in script
    assert "--prostate-label 5" in script
    assert "10.25919/45T8-P065" in script
    assert "git rev-parse HEAD" in script
    assert "nvidia-smi" in script
    assert "python train.py" in script
    assert "python predict.py" in script
    assert "set -euo pipefail" in script
