# Experiment Environment Record

The records below are observations of the environments that produced retained artifacts. They are not a claim that a local Windows environment reproduces Rangpur bit-for-bit.

| Experiment group | Evidence location | Platform / accelerator | Recorded software | Scope |
| --- | --- | --- | --- | --- |
| Main matched formal baseline 683675 | `artifacts/archives/683675_683676_formal_full.tar.gz:artifacts/683675_formal_baseline/environment.json` | Linux `4.18.0-553.148.1.el8_10`, NVIDIA A100-PCIE-40GB | PyTorch `2.13.0+cu130`, CUDA build `13.0` | Main result |
| Main matched formal improved 683676 | `artifacts/archives/683675_683676_formal_full.tar.gz:artifacts/683676_formal_improved/environment.json` | Same retained platform and GPU | PyTorch `2.13.0+cu130`, CUDA build `13.0` | Main result |
| Baseline-new 687753 | `artifacts/2026-10-23/687753_formal_baseline_new/environment.json` | Linux `4.18.0-553.148.1.el8_10`, NVIDIA A100-PCIE-40GB | PyTorch `2.13.0+cu130`, CUDA build `13.0` | Wide20 supplementary comparator |
| Recovered Wide20 687754 | `artifacts/2026-10-23/687754_formal_improved_wide20_recovery/environment.json` | Linux `4.18.0-553.148.1.el8_10`, NVIDIA A100-PCIE-40GB | PyTorch `2.13.0+cu130`, CUDA build `13.0` | Wide20 supplementary arm |
| Historical local snapshot | `docs/requirements-verified-local.txt` | Windows local Python environment | Exact importable package versions dated 2026-10-23 | Tests, plots and documentation only |

`requirements.txt` remains a portable range-based development specification. `docs/requirements-verified-local.txt` is a short, installation-verified local snapshot for the packages imported by this project; it is not a complete `pip freeze`, a lockfile, or evidence for the remote formal experiments.

The saved environment JSON records PyTorch/CUDA/GPU but does not include a complete remote package freeze. No undocumented package version is inferred for the Rangpur runs.

The README's PowerShell and Bash examples declare `--seed 42` and `--device` explicitly for a new, separately registered run. They do not reproduce historical remote environments bit-for-bit, and they must not be used to rerun held-out testing for configuration selection.

## Context v2 and current local verification

The local package for job `692354` is `artifacts/2026-10-25/692354_context_v2_ablation/context_v2_ablation_692354/`. `python_version.txt` records Python 3.11.15; `gpu.txt` records NVIDIA A100-PCIE-40GB and driver 590.48.01. The NVIDIA-SMI CUDA capability display is 13.1; this is not a recorded PyTorch CUDA build. No complete remote pip freeze or independent torch/CUDA-build record was found in this package, so historical formal PyTorch 2.13.0/CUDA 13.0 values must not be assigned to this run. `git_sha.txt` records `639e5ed212736d5ef789d148fd27f6ef2c50ff73`, accompanied by `source_changes.patch`; the SHA alone does not identify an unmodified checkout. Earlier `691652` is a separate run and does not inherit these facts.

The 2026-10-23 local snapshot above is historical evidence for its own imports and plotting. Current shell Python reports 3.11.9, distinct from remote Context Python 3.11.15; the final current test record must identify its actual interpreter, date, SHA and results. Do not manufacture historical remote versions using a local installation or new pip freeze.

For future runs, capture `python --version`, interpreter path, `torch.__version__`, `torch.version.cuda`, `python -m pip freeze`, NVIDIA driver/GPU details, Git SHA plus dirty diff, and manifest SHA-256 before execution. Save these alongside the actual command and final scheduler accounting. Missing historical environment evidence does not justify retraining or re-evaluating viewed test patients.

Restored from `docs/superpowers/specs/EXPERIMENT_ENVIRONMENT.md` and updated on 2026-10-25. This record separates evidence groups; it is not a lockfile or current suite-pass certificate.
