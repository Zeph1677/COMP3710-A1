# HipMRI Wide20 Matched Formal Protocol Design

## Decision

This protocol preserves all existing artefacts and treats the improved
validation sweep as candidate selection only.  It creates three independent
Rangpur jobs: a train/validation-only `candidate_00_control`, a new matched
baseline with a single held-out evaluation, and either a held-out evaluation
of the verified frozen `wide20` checkpoint or a replacement matched wide20
training job followed by one held-out evaluation.

## Immutable protocol

All new training uses commit `2e25298`, seed `42`, `prostate_label=5`, the
frozen CSIRO source, target shape `128 128 128`, train patch shape `96 96 96`,
foreground-patch probability `0.50`, CE/Dice weights `4.0/4.0`, learning rate
`0.001`, and validation checkpoint selection metric `prostate_dice`.  The
maximum training budget is 20 epochs; a selected epoch records a result and
does not alter that budget.

The control uses Improved `base_channels=16` and does not invoke prediction.
The matched baseline uses Baseline `base_channels=16`.  The wide20 arm uses
Improved `base_channels=20`.  A formal test command can run exactly once for
each selected formal model; it is forbidden for the control.

## Job boundary and output contract

Each job uses a new PID-first output directory under
`artifacts/rangpur_formal_wide20_matched_20261022/` and refuses a pre-existing
directory.  The job records Git state, frozen label source/hash, re-audited
manifest, command line, environment, GPU data, stdout/stderr, checkpoint hash,
history, summaries and final scheduler accounting.  Test arms also retain
test metrics and all per-case prediction tensors.

An account-collector command runs after a job exits.  It obtains `sacct -X`
with the required columns and accepts only `COMPLETED` with `0:0`; the
in-job EXIT snapshot is retained as diagnostic information but never accepted
as final accounting.

## Frozen wide20 recovery

Before an improved held-out test, recovery inspects the previous sweep
directory for `best.pt`, experiment configuration and history.  It verifies
the checkpoint checksum, commit SHA, manifest hash, label source,
`selection_metric=prostate_dice`, `selected_epoch=11`, and the required
wide20 configuration.  The recovery report is immutable.  A failed recovery
does not test the old artefact; it selects the explicit retraining path.

## Completion and comparison

Control completion is evaluated only on validation evidence.  The baseline and
wide20 artefacts must share manifest, held-out patient IDs, geometry, patch,
stride and label mapping before `compare_failures.py` runs.  The comparison
records five patient-level prostate Dice values, paired differences, patient
TP/FP/FN/precision/recall, supplementary non-background Dice/IoU, resource
fields, and aligned prostate FP/FN panels.  No parameter changes follow a
held-out result.

README and report updates are gated on both final accounting records and all
comparison artefacts.  The historical `683675` baseline and older result
directories remain unchanged and separately labelled.

