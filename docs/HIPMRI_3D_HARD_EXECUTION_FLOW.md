# HipMRI 3D Hard: reproduction, evidence and acceptance guide

This is the current guide for the flattened private repository. Source code is in `src/`, tests in `tests/`, Slurm scripts in `scripts/`, evidence in `evidence/`, and generated README assets in `docs/assets/`.

## Current evidence status

The main matched formal comparison is Git `4232e0d`, jobs 683675 (baseline) and 683676 (Improved). Both final accounting rows are `COMPLETED|0:0`; their test results are 0.811548 and 0.813587 patient-level prostate Dice. The supplementary Wide20 comparison is jobs 687753 and 687754, both `COMPLETED|0:0`; Baseline-new is 0.818316 and recovered Wide20 is 0.794166. Wide20 is a negative supplementary result, not future work and not the headline experiment.

Read [EXPERIMENT_EVIDENCE.md](EXPERIMENT_EVIDENCE.md) before quoting a result. It contains the retained paths, checksums, selected checkpoints, label source, scheduler evidence, and limitations. Use [EXPERIMENT_ENVIRONMENT.md](EXPERIMENT_ENVIRONMENT.md) to separate formal Rangpur environment facts from local test/plotting versions.

## Reproduce locally without changing experiment claims

```powershell
$dataRoot = Resolve-Path '.\data\HipMRI_Study_open'
python src\audit.py $dataRoot --output evidence\dataset_manifest.json --seed 42
python -m pytest -q
python -m compileall src
```

The audit regenerates a manifest and can change its hash if raw data or code differs. Do not overwrite a preserved formal artifact directory. Use a new ignored `artifacts/` output directory for local reproduction.

```powershell
python src\train.py --manifest evidence\dataset_manifest.json --data-root $dataRoot --output-dir artifacts\new_run --model baseline --epochs 20 --batch-size 1 --learning-rate 0.001 --base-channels 16 --target-shape 128 128 128 --train-patch-shape 96 96 96 --foreground-patch-probability 0.50 --prostate-label 5 --prostate-label-source 'CSIRO, Labelled weekly MR images of the male pelvis, v2, DOI: 10.25919/45T8-P065, BriefDataDescription.txt, Labels block: 5=Prostate.'
python src\predict.py --checkpoint artifacts\new_run\best.pt --manifest evidence\dataset_manifest.json --data-root $dataRoot --output-dir artifacts\new_test --patch-size 128 128 128 --stride 64 64 64
```

New output is a new experiment, not proof that it reproduces a retained Rangpur result. Do not select a configuration after reading held-out test metrics.

## Refresh README figures without training

```powershell
python src\plot_training.py --history artifacts\2026-10-22\683675_formal_baseline\training_summary.json --label Baseline --history artifacts\2026-10-22\683676_formal_improved\training_summary.json --label 'Improved 3D U-Net' --selection-metric mean_dice --output docs\assets\training_curves.png
python scripts\generate_readme_assets.py --paired-comparison-archive artifacts\archives\683675_683676_formal_paired_comparison.tar.gz
```

The first command reads histories only. The second extracts the already selected B037_Week0 paired panel byte-for-byte from the archive; it does not rerun inference or alter raw evidence.

## Acceptance checklist

- [ ] `python -m pytest -q` passes.
- [ ] `python -m compileall src` passes.
- [ ] `git diff --check` is clean.
- [ ] `wsl bash -n` passes for each `scripts/*.slurm` file.
- [ ] README images and local links resolve; the selected epochs on the curve match the corresponding summaries.
- [ ] Result tables recompute from the cited JSON artifacts, including the negative Wide20 difference.
- [ ] `git ls-files` contains no raw data, checkpoint, NIfTI, prediction tensor, archive, or `artifacts/` file.
- [ ] `git status --short` contains only intentional source, documentation, tracked figure, and expected course-PDF changes.
- [ ] A README PDF export, if a local renderer is available, is visually checked before submission.

## External assessment gates

The repository can provide code and local evidence only. Tutor feasibility check-off, GitHub pull request, staff code review response, and Turnitin PDF submission remain separate actions. A local commit, local PDF, job completion, or passing test suite does not prove any of those external actions occurred.
