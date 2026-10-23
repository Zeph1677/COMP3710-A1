# Experiment Environment Record

The records below are observations of the environments that produced retained artifacts. They are not a claim that a local Windows environment reproduces Rangpur bit-for-bit.

| Experiment group | Evidence location | Platform / accelerator | Recorded software | Scope |
| --- | --- | --- | --- | --- |
| Main matched formal baseline 683675 | `artifacts/archives/683675_683676_formal_full.tar.gz:artifacts/683675_formal_baseline/environment.json` | Linux `4.18.0-553.148.1.el8_10`, NVIDIA A100-PCIE-40GB | PyTorch `2.13.0+cu130`, CUDA build `13.0` | Main result |
| Main matched formal improved 683676 | `artifacts/archives/683675_683676_formal_full.tar.gz:artifacts/683676_formal_improved/environment.json` | Same retained platform and GPU | PyTorch `2.13.0+cu130`, CUDA build `13.0` | Main result |
| Baseline-new 687753 | `artifacts/2026-10-23/687753_formal_baseline_new/environment.json` | Linux `4.18.0-553.148.1.el8_10`, NVIDIA A100-PCIE-40GB | PyTorch `2.13.0+cu130`, CUDA build `13.0` | Wide20 supplementary comparator |
| Recovered Wide20 687754 | `artifacts/2026-10-23/687754_formal_improved_wide20_recovery/environment.json` | Linux `4.18.0-553.148.1.el8_10`, NVIDIA A100-PCIE-40GB | PyTorch `2.13.0+cu130`, CUDA build `13.0` | Wide20 supplementary arm |
| Current local verification | `docs/requirements-verified-local.txt` | Windows local Python environment | Exact importable package versions dated 2026-10-23 | Tests, plots and documentation only |

`requirements.txt` remains a portable range-based development specification. `docs/requirements-verified-local.txt` is a short, installation-verified local snapshot for the packages imported by this project; it is not a complete `pip freeze`, a lockfile, or evidence for the remote formal experiments.

The saved environment JSON records PyTorch/CUDA/GPU but does not include a complete remote package freeze. No undocumented package version is inferred for the Rangpur runs.
