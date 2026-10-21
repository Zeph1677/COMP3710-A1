# HipMRI Formal Comparison Evidence Design

## Goal

Turn the existing single-model smoke pipeline into a reproducible formal-comparison evidence pipeline. It must compare baseline and improved checkpoints at the same clinically relevant prostate location, retain class-specific error evidence, and capture scheduler accounting after a Slurm job has finished.

## Scope and evidence boundary

The active project is HipMRI 3D Hard only. It uses the fixed patient-disjoint manifest, `prostate_label=5`, and the frozen CSIRO mapping source. No OASIS artifact may enter the generated metrics, panels, feasibility material, or final report.

The existing one-epoch baseline and improved jobs remain smoke evidence. They prove real-data execution, not a performance claim, a formal comparison, or satisfaction of the course Dice target.

## Comparison panels

Add a comparison renderer that consumes two held-out predictions, one ground-truth label volume, and one MRI volume from the same case. It determines a prostate-relevant slice by first selecting the lowest non-empty prostate-label axial index (the prostate base under the project orientation convention); if the reference has no prostate voxels it fails rather than silently choosing the centre slice.

Each generated panel uses that exact axial slice and contains MRI, ground truth, baseline prediction, improved prediction, baseline prostate-only false-positive/false-negative map, and improved prostate-only false-positive/false-negative map. The maps use `prostate_label=5`: false positive is prediction equals 5 while ground truth is not 5; false negative is ground truth equals 5 while prediction is not 5. The file metadata records case ID, selected axial index, label index, both checkpoint paths, manifest hash, and each model's case Dice.

The renderer operates only after both runs have already produced test predictions. It does not select a different patient or slice independently for each model. Formal report selection ranks patients by the declared patient-level prostate Dice and renders three to five cases from the shared held-out set.

## Resource records

The existing train and prediction artifacts remain the authoritative source of measured values. Each new formal run records parameter count, training time, peak training VRAM, total and per-patient inference time, peak inference VRAM, target shape, patch size, stride, overlap, and the overlap-averaged-logit merge method. A comparison summary reads values from both model artifacts without recomputing metrics.

## Scheduler accounting

Slurm cannot reliably report final accounting state from inside the running job. Add a login-node collector that accepts a completed job ID and an output directory. It writes `sacct.txt` containing `JobID,JobName,State,ExitCode,Elapsed,AllocTRES`, rejects a state other than `COMPLETED` with exit code `0:0`, and copies the corresponding Slurm stdout/stderr files when present. The collector does not submit, cancel, or modify jobs.

## Formal-run protocol

The formal baseline and improved runs use the same manifest hash, model-independent preprocessing, target geometry, seed, epoch count, batch size, learning rate, patch size, stride, and prostate mapping source. Each model selects its checkpoint from validation only. Held-out test inference occurs once per selected checkpoint. The comparison pipeline consumes only these frozen artifacts.

## Testing

Tests must prove that the comparison renderer rejects mismatched shape or missing prostate reference, uses the same prostate-base slice for both models, and produces prostate-specific error masks rather than all-foreground masks. Tests must also prove the accounting collector rejects incomplete/non-zero-exit accounting output and writes an accepted completed record. Existing full-suite tests remain required.

## Documentation

Update the execution guide to reflect verified state: CSIRO mapping complete, 25 local tests at the time of this design, real A100 smoke artifacts available, and formal runs still outstanding. The guide must distinguish smoke metrics from formal metrics and point to the post-job collector for scheduler evidence.
