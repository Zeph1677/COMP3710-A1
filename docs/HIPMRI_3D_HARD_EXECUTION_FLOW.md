# HipMRI 3D Hard: reproduction, evidence and acceptance guide

This is the current guide for the flattened private repository. Source code is in `src/`, tests in `tests/`, Slurm scripts in `scripts/`, evidence in `evidence/`, and generated README assets in `docs/assets/`.

## Current evidence status

The main matched formal comparison is Git `4232e0d`, jobs 683675 (baseline) and 683676 (Improved). Both final accounting rows are `COMPLETED|0:0`; their test results are 0.811548 and 0.813587 patient-level prostate Dice. The supplementary Wide20 comparison is jobs 687753 and 687754, both `COMPLETED|0:0`; Baseline-new is 0.818316 and recovered Wide20 is 0.794166. Wide20 is a negative supplementary result, not future work and not the headline experiment. The final raw accounting for 687753/687754 is retained locally at `artifacts/2026-10-23/687753_687754_final_accounting/sacct_final.txt`; it supersedes the earlier running-time snapshot for terminal-state verification.

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
python src\train.py --manifest evidence\dataset_manifest.json --data-root $dataRoot --output-dir artifacts\new_run --model baseline --epochs 20 --batch-size 1 --learning-rate 0.001 --base-channels 16 --target-shape 128 128 128 --train-patch-shape 96 96 96 --foreground-patch-probability 0.50 --prostate-label 5 --prostate-ce-weight 4 --prostate-dice-weight 4 --selection-metric mean_dice --seed 42 --device cuda --prostate-label-source 'CSIRO, Labelled weekly MR images of the male pelvis, v2, DOI: 10.25919/45T8-P065, BriefDataDescription.txt, Labels block: 5=Prostate.'
python src\predict.py --checkpoint artifacts\new_run\best.pt --manifest evidence\dataset_manifest.json --data-root $dataRoot --output-dir artifacts\new_test --patch-size 128 128 128 --stride 64 64 64 --device cuda
```

New output is a new experiment, not proof that it reproduces a retained Rangpur result. Do not select a configuration after reading held-out test metrics.

## Refresh README figures without training

```powershell
python src\plot_training.py --history artifacts\2026-10-22\683675_formal_baseline\training_summary.json --label Baseline --history artifacts\2026-10-22\683676_formal_improved\training_summary.json --label 'Improved 3D U-Net' --selection-metric mean_dice --output docs\assets\training_curves.png
python scripts\generate_readme_assets.py --paired-comparison-archive artifacts\archives\683675_683676_formal_paired_comparison.tar.gz
```

The first command reads histories only. The second extracts the already selected B037_Week0 paired panel byte-for-byte from the archive; it does not rerun inference or alter raw evidence.

## Architecture identity and validation boundary

The primary `4232e0d` Improved implementation is `legacy_improved_v1` with 1,423,106 base16 parameters, transposed-convolution upsampling and training-only auxiliary outputs. A new Improved run must explicitly add `--model improved --architecture-version legacy_improved_v1` to select that historical architecture. The default new `--model improved` selects `improved_context_v2`; it has interpolation, dropout and inference-time multi-scale logit fusion and must not be labelled a rerun of the primary Improved arm. Prediction uses saved configuration via the model factory; historical Improved configurations without a version map to legacy. Use strict state loading, never `strict=False` to conceal a mismatch.

Context job `692354` is validation-only, and all candidate arms fail at least one original gate. B has the best validation Dice but 1.486× training memory exceeds 1.35×. See [the independent validation record](context_v2_ablation_results_20261025.md); do not run another held-out evaluation for this convergence work.

## Current acceptance checklist

The specs copy recorded 66 tests and a four-script syntax check on 2026-10-24. Those observations are historical and do not establish the current result. The following items require a fresh final record after all source, document and asset edits:

- [x] Run `python -m pytest -q`; record actual count, interpreter, date and SHA.
- [x] Run `python -m compileall -q src` and `git diff --check`.
- [x] Run `wsl bash -n` for all six current Slurm scripts using verified paths.
- [x] Verify old/new model strict round-trip loading and actual historical checkpoint loading if available; state missing evidence.
- [x] Verify README local links against retained tracked files or explicit pending inclusion list, selected epochs and result JSON.
- [x] Verify no raw data, checkpoint, NIfTI, prediction tensor, complete artifacts or archive is distributed.
- [x] Export a new dated internal README PDF and inspect every rendered page; October 23/24 PDFs do not verify this revision.
- [x] Rebuild a new dated course package without overwriting October 24 and inspect contents and links.

The AI disclosure body is intentionally empty for the author. Keep the PDF for internal review until the author fills it and a fresh export is checked. Final scheduler accounting for Context `692354` is a separate missing item; complete histories are not accounting output.

## External assessment gates

The repository can provide code and local evidence only. Tutor feasibility check-off, GitHub pull request, staff code review response, and Turnitin PDF submission remain separate actions. A local commit, local PDF, job completion, or passing test suite does not prove any of those external actions occurred.

Restored from `docs/superpowers/specs/HIPMRI_3D_HARD_EXECUTION_FLOW.md` and revised on 2026-10-25.


## Prior convergence verification — 25 October 2026

The checks above were completed against checkpoint commit `d22c291` plus the staged convergence changes, not against a clean historical checkout. Windows Python `E:/Python/python.exe` is 3.11.9; current torch is 2.11.0+cu128. The source suite reports **93 passed**. The fresh flattened course export reports **92 passed, 1 skipped**: its local-export-helper test is intentionally skipped because that helper is not distributed. Source compileall, Git diff checks and LF/WSL Bash syntax checks for all six Slurm scripts passed. README and actual exported Markdown links resolve to included files; no raw data/checkpoints/complete artifacts or full animation gallery is distributed.

Actual historical 683676 and Context B weights load strictly and produce exactly equal synthetic-input eval logits against the respective original implementations. The new internal README PDF has 10 rendered and visually reviewed pages, including a standalone feasibility review on page 4. AI disclosure remains blank and the PDF is marked INTERNAL REVIEW on every page. The fresh course export and ZIP are local review artifacts; they are not a submitted PR or Turnitin receipt. Detailed hashes and QA records are local `local/project-convergence-20261025/`.

The only attempted external evidence retrieval was a read-only final accounting query for 692354; SSH timed out. That gap remains explicitly open. No training or new held-out evaluation was performed during convergence.

## Restoration with animation retention

The current restoration is based on main 5d4fabc plus the restored convergence changes. The author explicitly requested preserving GIF display, superseding the prior two-panel distribution policy. The reviewed GIFs and gallery are now included by exact filename whitelist in the repository and course export, while MRI, checkpoints and artifacts stay excluded. The fresh source suite reports 94 passed; the actual exported package result is recorded under local/restore-with-gifs-20261025/. Earlier test/PDF/package counts above describe the prior convergence snapshot. AI disclosure remains blank; no new training or held-out inference was performed.
