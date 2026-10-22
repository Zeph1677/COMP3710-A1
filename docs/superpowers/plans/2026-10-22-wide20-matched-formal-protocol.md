# HipMRI Wide20 Matched Formal Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and execute a non-overwriting, validation-selected matched Baseline-new versus Improved-wide20 protocol.

**Architecture:** A dedicated Slurm script owns the three job modes and their immutable artifact contract. A Python recovery/collection utility verifies the old wide20 checkpoint and final Slurm accounting before any formal comparison consumes results.

**Tech Stack:** Python 3, PyTorch, Bash/Slurm, JSON, SHA-256, Pytest.

**Spec:** `docs/superpowers/specs/2026-10-22-wide20-matched-formal-protocol-design.md`

## Global Constraints

- Do not alter, delete, or reuse historical result directories as new formal outputs.
- Control mode never invokes `predict.py` and never reads the test partition.
- Baseline-new and Improved-wide20 use `prostate_dice` validation selection and execute held-out prediction once.
- Formal test output requires final `sacct -X` evidence showing `COMPLETED` and `0:0`.
- Do not push or create a PR. Do not update performance claims until fresh artefacts satisfy the protocol.

---

### Task 1: Formal job-mode contract

**Files:**
- Create: `scripts/run_hipmri_wide20_matched_formal.slurm`
- Modify: `tests/test_rangpur_smoke.py`

**Interfaces:**
- Environment: `HIPMRI_FORMAL_MODE` in `control`, `baseline_new`, `improved_wide20`.
- Output: `artifacts/rangpur_formal_wide20_matched_20261022/<JOBID>_<mode>/`.

- [ ] **Step 1: Write failing static tests** for the three permitted modes, fixed protocol arguments, no prediction command in control, `--selection-metric prostate_dice`, PID-first path, non-overwrite guard, and retained required files.
- [ ] **Step 2: Run RED** with `python -m pytest tests/test_rangpur_smoke.py -q` and confirm failure because the new script does not exist.
- [ ] **Step 3: Implement the smallest dedicated Slurm script** that validates mode/configuration, re-audits the manifest, trains with the fixed configuration, stores checkpoint SHA-256, and invokes full-volume prediction only for the two formal arms.
- [ ] **Step 4: Run GREEN** with `wsl bash -n scripts/run_hipmri_wide20_matched_formal.slurm` and `python -m pytest tests/test_rangpur_smoke.py -q`.

### Task 2: Recovery and post-job accounting gates

**Files:**
- Create: `src/formal_wide20_protocol.py`
- Create: `tests/test_formal_wide20_protocol.py`

**Interfaces:**
- `verify_wide20_checkpoint(run_dir: Path, expected_sha: str) -> dict[str, object]`.
- `validate_final_sacct(text: str, job_id: str) -> dict[str, str]`.

- [ ] **Step 1: Write failing tests** for valid frozen wide20 metadata, every mismatching identity field, checkpoint checksum mismatch, `RUNNING|0:0`, failed exit code, and accepted completed accounting.
- [ ] **Step 2: Run RED** with `python -m pytest tests/test_formal_wide20_protocol.py -q`.
- [ ] **Step 3: Implement validation without remote side effects**; it reads local copied artefacts and `sacct -X` text, emitting a JSON report only after all checks pass.
- [ ] **Step 4: Run GREEN** with `python -m pytest tests/test_formal_wide20_protocol.py -q`.

### Task 3: Integrate and verify locally

**Files:**
- Modify: `README.md`
- Modify: `docs/HIPMRI_3D_HARD_EXECUTION_FLOW.md`

- [ ] **Step 1: Add protocol status wording** that calls wide20 a validation-selected candidate and states the control, matched pair, accounting and panel gates.
- [ ] **Step 2: Run full verification**: `python -m pytest -q`, `wsl bash -n scripts/run_hipmri_wide20_matched_formal.slurm`, and `git diff --check`.
- [ ] **Step 3: Commit only source, tests and documentation** with a local commit; do not stage data, checkpoints, raw NIfTI or result artefacts.

### Task 4: Rangpur staged execution and evidence retrieval

**Files:**
- Read/write ignored: `artifacts/rangpur_formal_wide20_matched_20261022/`

- [ ] **Step 1: Synchronize the committed source and check the previous wide20 directory.** Copy a recovered checkpoint only after checksum/identity verification. If recovery fails, record the cause and select retraining mode.
- [ ] **Step 2: Submit control mode.** Require final `sacct -X` `COMPLETED|0:0`, control-only artefacts, and a valid validation summary before continuing.
- [ ] **Step 3: Submit matched Baseline-new and the valid wide20 route.** Preserve exact job IDs and do not edit protocol parameters after submission.
- [ ] **Step 4: Retrieve all required artifacts and final accounting.** Verify files, hashes, metadata and five test patients; run `compare_failures.py` on the matched pair.
- [ ] **Step 5: Update the independent results section only when both formal arms and paired panels meet all gates.**

## Plan self-review

Task 1 establishes immutable Slurm execution, Task 2 establishes recovery/accounting acceptance, Task 3 provides regression and documentation evidence, and Task 4 governs external execution without altering historical results. The test interface names, output path and job modes match across all tasks. No task permits test access from control mode or formal result claims without final accounting.
