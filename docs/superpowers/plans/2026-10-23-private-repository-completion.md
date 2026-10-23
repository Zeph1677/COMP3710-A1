# Private Repository Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn this private HipMRI repository into an auditable completion package whose code, results, figures, and documentation make only claims supported by locally retained evidence.

**Architecture:** Keep raw data, checkpoints, prediction tensors, and downloaded execution artifacts ignored. Add a small reusable plotting module that consumes saved JSON histories, generate only compact README assets, and place the audit trail in Markdown documents that point to immutable local artifact paths and checksums. The README presents `4232e0d` as the main matched formal experiment and the Wide20 comparison as a negative supplementary result.

**Tech Stack:** Python 3, PyTorch, Matplotlib, pytest, PowerShell, Bash syntax checking, Git.

**Spec:** `C:\Users\ASUS\.codex\attachments\eb6594a9-5c65-4571-a9e8-39ef1814de7b\已粘贴的文本.txt`

## Global Constraints

- Work directly on the current repository branch; do not create a worktree, push, create a PR, retrain, or submit a new Slurm job.
- Preserve the pre-existing replacement of `docs/COMP3710_Report_2026_draft_v2.4.pdf` with `docs/COMP3710_Report_2026_v2.5.pdf`.
- Do not commit datasets, checkpoints, NIfTI volumes, prediction tensors, archives, or `artifacts/` contents.
- State `5=Prostate` only with the cited CSIRO source and preserve the manifest SHA in result claims.
- Do not claim that a history has prostate-specific validation metrics when its saved history does not contain them.
- Run fresh tests, compilation, diff, Bash, image, link, and repository-status checks before the completion commit.

---

### Task 1: Evidence inventory and final accounting boundary

**Files:**
- Create: `docs/EXPERIMENT_EVIDENCE.md`
- Create: `docs/EXPERIMENT_ENVIRONMENT.md`
- Create: `docs/requirements-verified-local.txt`

**Interfaces:**
- Consumes: retained JSON/log/checksum files beneath ignored `artifacts/` and `evidence/dataset_manifest.json`.
- Produces: a path-level evidence index, result-status distinction, and non-portable exact local dependency snapshot.

- [x] Record the main formal pair (`4232e0d`, jobs 683675/683676), its manifest hash, configuration, selected epochs, test protocol, metrics, and B037 panel locations.
- [x] Record the five-candidate validation scan and Wide20 recovery/control lineage separately from the main experiment.
- [x] Query final scheduler accounting for jobs 687753/687754 without starting work; retain the result as audit evidence if available.
- [x] Record Baseline-new/Wide20 result values and the negative difference only after comparison of their protocol fields.
- [x] Distinguish retained exact runtime observations from an installable dependency specification; do not project the 2026-10-23 environment onto 683675/683676 where no exact retained record exists.

### Task 2: Saved-history training-curve renderer

**Files:**
- Create: `tests/test_plot_training.py`
- Create: `src/plot_training.py`
- Modify: `src/train.py`

**Interfaces:**
- Produces: `load_history(path: Path) -> list[dict[str, object]]`, `selected_epoch(history, selection_metric) -> int`, and `render_training_curves(series, output_path) -> None`.
- Consumes: JSON history lists or `training_summary.json` objects with a `history` list.

- [x] Write tests that make missing module/import fail: valid history yields its selected epoch; `mean_dice` selects the largest `validation_mean_dice`; prostate selection rejects a history without complete prostate fields; rendering creates a non-empty PNG.
- [x] Run the dedicated test file and confirm the expected missing-module failure.
- [x] Implement a history-only renderer with two panels: train loss and declared validation selection metric; mark selected epochs and label unavailable prostate metric rather than inventing it.
- [x] Add an explicit CLI for one or more `--history` inputs, `--selection-metric`, `--label`, and `--output`; make `train.py` render into the run output after a live run.
- [x] Run the dedicated tests again and generate `docs/assets/training_curves.png` from retained main-experiment histories without retraining.

### Task 3: README visual assets and evidence-safe documentation

**Files:**
- Create: `scripts/generate_readme_assets.py`
- Create: `docs/assets/model_architecture.png`
- Create: `docs/assets/failure_case_b037.png`
- Modify: `README.md`
- Modify: `docs/HIPMRI_3D_HARD_EXECUTION_FLOW.md`

**Interfaces:**
- Consumes: `src/modules.py`, result JSON files, saved histories, and the existing B037 main-experiment panels.
- Produces: deterministic small PNG assets and a README with direct relative links.

- [x] Render an architecture image that reflects actual modules: three encoder reductions, bottleneck, three decoder expansions with skip concatenation; residual blocks and two decoder auxiliary heads only for Improved U-Net; final head only at inference; no attention-gate claim.
- [x] Copy or regenerate the paired main-experiment B037 comparison asset without altering the raw evidence; caption its error-relevant prostate-range slice, FP/FN legend, SHA, and selection rule.
- [x] Rewrite README sections in the required result order and explain tensor shapes, preprocessing, weighted CE/Dice plus auxiliary loss, foreground patches, validation selection, averaged-logit overlap reconstruction, and patient-level headline scoring.
- [x] Include main matched results, class-wise supplementary outcomes, resource profile, five failure cases, actionable recommendation, supplementary Wide20 limitation, exact AI disclosure, usage commands, and evidence index.
- [x] Replace obsolete execution-flow claims and commands with the current flattened paths and an evidence/reproduce/acceptance workflow.

### Task 4: Acceptance verification and local completion commit

**Files:**
- Modify: tracked completion assets and documents only.

- [x] Run `pytest -q`, `python -m compileall src`, `git diff --check`, and `wsl bash -n` on every Slurm script.
- [x] Check generated PNG dimensions/readability, README local links, result-table calculations, selected epochs, and the absence of prohibited tracked data/checkpoint/artifact files.
- [x] Export the README to PDF if a local renderer is available and visually inspect the rendered pages; record any unavailable renderer fact rather than claiming a visual export.
- [x] Review `git status` and staged diff to ensure the expected v2.4 deletion/v2.5 addition are retained and no ignored raw artifacts are staged.
- [x] Create one local completion commit; do not push or create a PR.
