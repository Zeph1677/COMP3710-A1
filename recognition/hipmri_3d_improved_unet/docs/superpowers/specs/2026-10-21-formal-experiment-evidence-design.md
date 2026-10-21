# HipMRI Formal Experiment Evidence Design

## Purpose and decision

The completed Rangpur jobs `680366` (baseline) and `680381` (improved) are
real-data, one-epoch smoke runs. They verify the execution chain, not model
quality: both held-out mean patient prostate Dice values are `0.0`. This design
prepares a reproducible formal comparison without promoting smoke values into
the report's performance table.

The formal comparison will use the existing patient-disjoint manifest, the
frozen CSIRO mapping `5=Prostate`, the same target geometry, the same patch and
stride, and the same validation checkpoint-selection rule for both models.
Only each selected checkpoint's test-only prediction command may read the
held-out test split.

## Evidence contract

Each formal training directory will contain `training_summary.json` with:

- model name, parameter count, selected epoch, training duration and per-epoch
  training/validation duration;
- peak CUDA allocated memory for the training and validation phases, expressed
  in MiB and `0.0` for a non-CUDA run;
- the immutable experiment configuration already saved by the project,
  including manifest SHA-256, prostate label and cited source.

Each formal prediction directory will contain `test_metrics.json` with:

- the existing patient-level headline prostate Dice, patient count, per-class
  Dice/IoU and low-to-high prostate-Dice ranking;
- wall-clock total inference time, inference time for every held-out patient,
  mean per-patient inference time and peak CUDA allocated memory in MiB;
- checkpoint path, manifest SHA-256, prostate label source, target geometry,
  patch size, stride, overlap derived from patch/stride, and the declared
  `overlap_averaged_logits` merge method.

The timing boundary for prediction starts immediately before the first
patient's full-volume sliding-window inference and ends after the last
patient's prediction. Loading NIfTI files and writing panels are excluded from
the reported inference timing. CUDA work is synchronized at timing boundaries
when CUDA is in use.

## Comparative failure panels

The current single-model panels remain smoke evidence. A new comparison
renderer will consume one MRI volume, one ground-truth mask, and baseline and
improved predictions for the same case. It will render one axial comparison:

`MRI | Ground Truth | Baseline | Improved | Baseline prostate FP/FN | Improved prostate FP/FN`.

The selected axial index is deterministic. It is the highest axial index that
contains prostate in the ground truth (the prostate base in the project array
convention). If no ground-truth prostate voxel exists, rendering fails rather
than silently selecting a volume-centre slice. The FP/FN panels compare only
the frozen prostate class: false positive is prediction equals prostate while
ground truth does not, and false negative is ground truth equals prostate while
prediction does not.

The comparison command will accept both formal test-metrics directories and
their saved per-volume predictions. It will reject a baseline/improved pair
whose manifest SHA-256, test patients, prostate label, source text, patch size,
stride, or target geometry differs. It ranks patients by the mean of the two
models' patient-level prostate Dice, chooses the requested lowest-scoring cases,
and writes a JSON index next to the panels. No comparison panel is generated
from a smoke checkpoint for the final report.

## Train-only overfit preflight

Before formal training, a dedicated preflight command will construct a
train-only manifest view for an explicit patient ID. The patient must belong to
the existing `train` split; validation and test patients are rejected. The
command trains on the requested patient's first manifest case for a configured
number of epochs, evaluates that same case after each epoch, and writes its
result to an isolated output directory marked `preflight_only`.

This is a diagnostic gate, not a performance experiment and not a checkpoint
candidate. It must not invoke `predict.py`, calculate test metrics, or write a
headline prostate Dice. A failed preflight blocks the formal Slurm submission;
a passed preflight demonstrates only that the selected pipeline can fit one
training example.

## Slurm and report evidence

The formal Slurm job will save `sacct` output for its own job ID after training
and prediction complete. The job script will expose only safe formal parameters
(model name, epoch count and output root) and will reject the preflight mode.
It continues to re-audit public Rangpur data, compare the regenerated manifest
with the committed one, record Git SHA/status, capture the environment and
preserve stdout/stderr.

README and the execution guide will gain a clearly labelled smoke-evidence
table: scheduler state, elapsed time, A100 40 GB, PyTorch/CUDA environment,
one epoch, and the statement that `0.0` patient prostate Dice is not a formal
result. The formal-result table remains empty until a matched pair of completed
formal jobs creates its own immutable metrics artifacts. The report discussion
will cite the final comparative panels and resource fields, never the smoke
values as clinical performance.

## Tests and non-negotiable safeguards

- Unit tests prove training summary fields are populated and CPU runs report
  `0.0` CUDA memory.
- Synthetic end-to-end prediction tests prove timing/resource/config fields
  are saved without replacing patient-level Dice with pooled-voxel Dice.
- Renderer tests prove the chosen slice is the prostate-base index and that
  FP/FN is prostate-only.
- Pair-validation tests reject incompatible model artifacts before panel output.
- Preflight tests reject validation/test patients and ensure its artifacts are
  marked non-reportable.
- Slurm static tests require `sacct` collection and prohibit passing a
  preflight option to a formal job.

## Commit sequence

1. Record formal training and prediction resource evidence.
2. Add train-only overfit preflight.
3. Add aligned baseline/improved prostate failure panels.
4. Align README, execution guide and formal Slurm evidence collection.
5. Run the full test suite, commit each tested unit, then submit matched
   baseline and improved 20-epoch Rangpur jobs only after the preflight passes.
