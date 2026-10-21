# HipMRI Formal Experiment Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce auditable formal HipMRI baseline/improved experiment artifacts, prostate-specific comparison panels, and a safe Rangpur workflow without treating one-epoch smoke data as performance results.

**Architecture:** `train.py` records phase timing and CUDA memory. `predict.py` records full-volume resource data and saves per-volume predictions. A comparison command reads two compatible prediction directories. A separate train-only preflight verifies fit behavior without reading validation/test data.

**Tech Stack:** Python 3, PyTorch, Nibabel, Matplotlib, Pytest, Bash/Slurm, JSON, Git.

**Spec:** `docs/superpowers/specs/2026-10-21-formal-experiment-evidence-design.md`

## Global Constraints

- Formal baseline/improved runs use the same manifest, label/source, geometry, patch, stride, seed, optimizer settings and validation checkpoint rule.
- Test patients are read only by `predict.py` after a validation-selected `best.pt`; preflight never calculates test metrics or invokes `predict.py`.
- A missing prostate reference mask is an error for comparison rendering; a centre-slice fallback is forbidden.
- CUDA timing synchronizes at timing boundaries. CPU peak-memory fields equal `0.0` MiB.
- Smoke jobs `680366` and `680381` remain execution evidence, never formal performance evidence.
- Do not push, open a PR, or claim target attainment without later authorization and fresh formal-artifact verification.

---

### Task 1: Persist training phase resources

**Files:**
- Modify: `train.py:100-211`
- Modify: `tests/test_train.py:23-151`

**Interfaces:**
- History fields: `train_seconds`, `validation_seconds`, `train_peak_vram_mb`, `validation_peak_vram_mb`.
- Summary fields: `training_seconds`, `parameter_count`, `peak_train_vram_mb`, `peak_validation_vram_mb`.

- [ ] **Step 1: Write a failing resource-artifact test**

```python
def test_run_training_records_phase_resources_in_history_and_summary(tmp_path: Path):
    summary, output = synthetic_training_run(tmp_path, device="cpu")
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    saved = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
    assert {"train_seconds", "validation_seconds", "train_peak_vram_mb", "validation_peak_vram_mb"} <= set(history[0])
    assert history[0]["train_peak_vram_mb"] == 0.0
    assert history[0]["validation_peak_vram_mb"] == 0.0
    assert {"peak_train_vram_mb", "peak_validation_vram_mb"} <= set(saved)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_train.py::test_run_training_records_phase_resources_in_history_and_summary -q`

Expected: FAIL because the fields do not exist.

- [ ] **Step 3: Implement minimal collection**

Add `_reset_cuda_peak(device)` and `_cuda_peak_mb(device)`. Reset before each
training/validation phase, time phases independently, record all four fields
per epoch, and derive summary maxima from history. Preserve validation-only
checkpoint selection.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_train.py -q; python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run: `git add train.py tests/test_train.py && git commit -m "feat: record HipMRI training resource evidence"`

### Task 2: Persist prediction resources and per-volume outputs

**Files:**
- Modify: `predict.py:41-136`
- Modify: `tests/test_predict.py:92-124`

**Interfaces:**
- `predictions/<case_id>.pt` stores CPU `torch.long` prediction, `case_id`, `patient_id`, `manifest_sha256`, `prostate_label`, and `target_shape`.
- `test_metrics.json` adds `total_inference_seconds`, `per_patient_inference_seconds`, `mean_patient_inference_seconds`, `peak_inference_vram_mb`, `target_shape`, `overlap`, and `logit_merge_method="overlap_averaged_logits"`.

- [ ] **Step 1: Write a failing end-to-end test**

```python
def test_run_prediction_records_resources_and_case_output(tmp_path: Path):
    output, _ = synthetic_prediction_run(tmp_path, device="cpu")
    saved = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
    assert saved["peak_inference_vram_mb"] == 0.0
    assert saved["total_inference_seconds"] >= 0.0
    assert saved["mean_patient_inference_seconds"] >= 0.0
    assert saved["overlap"] == [0.5, 0.5, 0.5]
    assert saved["logit_merge_method"] == "overlap_averaged_logits"
    assert (output / "predictions" / "B003_Week0.pt").is_file()
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_predict.py::test_run_prediction_records_resources_and_case_output -q`

Expected: FAIL because resource fields and predictions do not exist.

- [ ] **Step 3: Implement minimal collection**

Time only full-volume `sliding_window_logits`, synchronize CUDA at timing
boundaries, sum weekly-volume times per patient, save every segmentation, and
reject strides greater than matching patch dimensions before calculating overlap.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_predict.py -q; python -m pytest -q`

Expected: all tests pass, including test-patient isolation.

- [ ] **Step 5: Commit**

Run: `git add predict.py tests/test_predict.py && git commit -m "feat: record HipMRI inference resource evidence"`

### Task 3: Add train-only single-case overfit preflight

**Files:**
- Modify: `train.py:135-219`
- Modify: `tests/test_train.py`

**Interfaces:**
- `run_overfit_preflight(args) -> dict[str, object]` and CLI `--preflight-patient-id`.
- `preflight_summary.json` has `preflight_only=true`, patient/case IDs, epochs, history and `final_train_case_prostate_dice`.
- Reject non-train patients; never write `best.pt`, `training_summary.json`, or test metrics.

- [ ] **Step 1: Write failing isolation tests**

```python
def test_preflight_rejects_validation_and_test_patients(tmp_path: Path):
    with pytest.raises(ValueError, match="train split"):
        run_overfit_preflight(synthetic_preflight_args(tmp_path, patient_id="C007"))

def test_preflight_writes_non_reportable_train_only_artifact(tmp_path: Path):
    output, _ = run_synthetic_preflight(tmp_path, patient_id="B006")
    saved = json.loads((output / "preflight_summary.json").read_text(encoding="utf-8"))
    assert saved["preflight_only"] is True
    assert not (output / "best.pt").exists()
    assert not (output / "training_summary.json").exists()
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_train.py -k preflight -q`

Expected: FAIL because no preflight interface exists.

- [ ] **Step 3: Implement isolated preflight**

Select the first manifest case for the named train patient. Reuse model and loss
primitives, but construct both loaders from only that case. Dispatch from
`main()` only when `--preflight-patient-id` is supplied.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_train.py -q; python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run: `git add train.py tests/test_train.py && git commit -m "feat: add train-only HipMRI overfit preflight"`

### Task 4: Render strict baseline/improved prostate-base comparisons

**Files:**
- Create: `compare_failures.py`
- Create: `tests/test_compare_failures.py`
- Modify: `tests/test_predict.py`

**Interfaces:**
- `validate_comparison_inputs(baseline_metrics, improved_metrics)`, `prostate_base_index(target, prostate_label)`, `render_comparative_prostate_panel(...)`, and CLI `compare_failures.py --baseline-dir --improved-dir --output-dir --max-cases`.
- Output: `comparison_index.json`, `comparison_panels/<patient_id>__<case_id>.png`.
- Compatible manifest hash, prostate label/source, patch/stride/target geometry and test patient IDs are required.

- [ ] **Step 1: Write failing renderer/validation tests**

```python
def test_prostate_base_index_selects_highest_reference_slice():
    target = torch.zeros(6, 4, 4, dtype=torch.long)
    target[1, 1:3, 1:3] = 5
    target[4, 1:3, 1:3] = 5
    assert prostate_base_index(target, prostate_label=5) == 4

def test_validate_comparison_inputs_rejects_mismatched_manifest_hash():
    with pytest.raises(ValueError, match="manifest_sha256"):
        validate_comparison_inputs({"manifest_sha256": "a", **compatible_metrics()}, {"manifest_sha256": "b", **compatible_metrics()})
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_compare_failures.py -q`

Expected: FAIL because the comparison module does not exist.

- [ ] **Step 3: Implement validation and six-panel renderer**

Load both metrics files and matching `predictions/*.pt`; require metadata/case
identity equality. Rank patients by mean model Dice, choose their lowest-mean
volume, select highest axis-0 prostate voxel index, then render MRI, GT,
baseline, improved, baseline prostate FP/FN, and improved prostate FP/FN.
Encode FP as `1`, FN as `2`, and fail on missing prostate reference.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_compare_failures.py tests/test_predict.py -q; python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run: `git add compare_failures.py tests/test_compare_failures.py tests/test_predict.py && git commit -m "feat: compare HipMRI prostate-base failure panels"`

### Task 5: Make Rangpur formal evidence and documents consistent

**Files:**
- Modify: `run_hipmri_smoke.slurm`
- Modify: `tests/test_rangpur_smoke.py`
- Modify: `README.md`
- Modify: `docs/HIPMRI_3D_HARD_EXECUTION_FLOW.md`

**Interfaces:**
- Environment: `HIPMRI_EPOCHS` positive integer defaulting to `20`; `HIPMRI_RUN_KIND` equal to `formal`.
- Artifact: `$RUN_DIR/sacct.txt` from `sacct -P -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES`.
- Reject invalid model/run-kind/epoch values and never pass a preflight flag.

- [ ] **Step 1: Write failing Slurm evidence test**

```python
def test_formal_script_records_own_scheduler_accounting():
    script = SLURM_SCRIPT.read_text(encoding="utf-8")
    assert 'HIPMRI_EPOCHS="${HIPMRI_EPOCHS:-20}"' in script
    assert 'HIPMRI_RUN_KIND="${HIPMRI_RUN_KIND:-formal}"' in script
    assert 'sacct -P -j "$JOB_ID"' in script
    assert 'State,ExitCode,Elapsed,AllocTRES' in script
    assert "--preflight-patient-id" not in script
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_rangpur_smoke.py -q`

Expected: FAIL because scheduler accounting and formal parameters are absent.

- [ ] **Step 3: Implement formal Slurm/documentation changes**

Rename job/artifact prefix to `formal`, validate environment variables, use
`HIPMRI_EPOCHS`, and collect scheduler output via a trap without masking the
actual job status. Copy root Slurm files only when present. Add a labelled
smoke-evidence table, correct label/test status, state the two-level
patient/weekly-volume mean, and keep formal performance cells empty.

- [ ] **Step 4: Verify GREEN**

Run: `bash -n run_hipmri_smoke.slurm; python -m pytest tests/test_rangpur_smoke.py -q; python -m pytest -q; git diff --check`

Expected: valid Bash, all tests pass, no whitespace errors.

- [ ] **Step 5: Commit**

Run: `git add run_hipmri_smoke.slurm tests/test_rangpur_smoke.py README.md docs/HIPMRI_3D_HARD_EXECUTION_FLOW.md && git commit -m "ops: prepare auditable HipMRI formal Rangpur runs"`

### Task 6: Gate and submit matched formal Rangpur jobs

**Files:**
- Read: `docs/superpowers/specs/2026-10-21-formal-experiment-evidence-design.md`
- Read: `docs/superpowers/plans/2026-10-21-formal-experiment-evidence.md`

**Interfaces:**
- Consumes completed preflight and clean committed source.
- Produces independent baseline/improved job IDs and downloaded ignored artifacts; no source commit unless an independently tested defect is found.

- [ ] **Step 1: Submit and verify train-only preflight**

Submit `train.py --preflight-patient-id <first-train-patient>` in a separate
Slurm job. Require `sacct` `COMPLETED|0:0`, `preflight_only=true`, no `best.pt`,
and finite final train-case prostate Dice.

- [ ] **Step 2: Submit matched formal jobs**

```bash
HIPMRI_MODEL=baseline HIPMRI_EPOCHS=20 HIPMRI_RUN_KIND=formal sbatch run_hipmri_smoke.slurm
HIPMRI_MODEL=improved HIPMRI_EPOCHS=20 HIPMRI_RUN_KIND=formal sbatch run_hipmri_smoke.slurm
```

Use two clean checkouts at the same commit and do not alter either protocol
from test output.

- [ ] **Step 3: Verify/download and create comparison evidence**

Require `sacct.txt` to show `COMPLETED|0:0`; verify Git/manifest/source
identity, resource fields, checkpoints, histories, metrics and predictions;
download into ignored dated artifacts; run `compare_failures.py`; inspect JSON
and images; only then update the formal README result values. Do not commit
checkpoints or patient data.

## Plan self-review

- Tasks 1-2 implement the resource contract, Task 3 implements the train-only gate, Task 4 produces aligned prostate-specific panels, Task 5 records scheduler evidence and corrects documentation, and Task 6 sets the external execution gate.
- This plan contains no unresolved placeholders or unspecified test commands.
- `predict.py` produces artifacts, `compare_failures.py` consumes them, and only `train.py` owns preflight; the formal Slurm script cannot enter preflight mode.
