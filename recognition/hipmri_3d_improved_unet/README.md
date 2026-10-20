# HipMRI 3D Improved U Net for Prostate Segmentation

## Scope

This is the COMP3710 Hard segmentation project. It uses downsampled HipMRI 3D NIfTI volumes from `/home/groups/comp3710/HipMRI_Study_open` to test whether volumetric residual modelling and deep supervision reduce clinically important prostate boundary and cross-slice errors enough to justify their compute cost.

The implementation compares a plain 3D U-Net with a 3D Improved U-Net. The improved model contains volumetric residual units and two decoder deep-supervision heads. This is distinct from the repository's OASIS 2D work and must not reuse its reported values.

## Data and validity controls

`dataset.py` pairs `semantic_MRs` and `semantic_labels_only` by their relative NIfTI paths. It validates 3D shape, affine agreement, finite intensity values and integral labels. Splits are by patient, not slice or patch, and the exact seed-42 manifest must be reused by both models.

No HipMRI data, checkpoints or final results are committed. The code has been checked only against synthetic NIfTI volumes locally; real performance, the prostate label index, memory use and image orientation must be audited on Rangpur before reporting results.

### Local Windows data root

For local audit and development, the ignored raw 3D dataset is stored at `E:\vscode\PatternAnalysis-2026\data\HipMRI_Study_open`. Its direct children must be `semantic_MRs` and `semantic_labels_only`; do not rename individual NIfTI files or commit this directory. From this project directory, run:

```powershell
$dataRoot = Resolve-Path "..\..\data\HipMRI_Study_open"
python audit.py $dataRoot --output evidence\dataset_manifest.json --seed 42
```

`audit.py` receives the data root as an argument rather than embedding a local path in Python. On Rangpur, pass `/home/groups/comp3710/HipMRI_Study_open` to the same command. This preserves one code path across local audit and formal GPU training.

## Confirmed evaluation protocol (staff reply, 2026-10-21)

The plain 3D U-Net is the baseline. Both models must use the same patient-level split, preprocessing, downsampling and evaluation protocol. The 211 volumes come from only 38 patients according to the staff reply, so neither volumes, slices nor patches may be split independently across train, validation and test.

The headline result is **prostate Dice**. For each held-out test patient, reconstruct the complete volume from its sliding-window logits, convert the reconstructed logits to the final segmentation, and compute Dice for the prostate label. Report the arithmetic mean of those patient-level prostate Dice values. The `>= 0.70` target applies to this value; it is not a pooled-voxel Dice, foreground macro Dice or a mean over every label.

Report Dice for every non-background semantic label as supplementary class-wise results. Do not report background Dice. A formal run must record its patch size, overlap, logit-merging method and the fact that metrics were computed only after full-volume reconstruction. Failure-case discussion must prioritise the prostate base.

The staff reply does **not** identify which of the observed labels `0`–`5` is prostate. Before a training or final-evaluation run, identify that index from the HipMRI reference paper or authoritative dataset documentation, record the citation and mapping in the immutable experiment configuration, and use the same mapping for both models. Do not infer the index from label frequency, mask size or label numbering.

## Feasibility review

**Need and scope.** The intended user is a radiotherapy contouring specialist. The prototype drafts a 3D segmentation for review; it is not an autonomous clinical decision system.

**Acceptance criteria.** Zero patient overlap; identical test patients for both models; mean per-held-out-patient prostate Dice at least 0.70 after full-volume reconstruction; Dice/IoU for every non-background label, peak VRAM and per-case inference time; 3-5 real failure cases with prostate base as the priority.

**Preliminary evidence.** Local synthetic-NIfTI tests validate recursive pairing, missing-pair rejection, shape/affine checks, patient splitting, 3D forward/backward passes, deep supervision and held-out metric fields. This is engineering evidence, not a trained-model result.

**Risks, budget and fallback.** 3D memory demand, small foreground regions, NIfTI orientation, slow I/O and failed cluster jobs require a short GPU smoke run before full Slurm training. If the improved model does not improve real held-out boundary errors at acceptable cost, report the plain 3D U-Net as the recommended fallback.

## Current verification

```powershell
cd E:\vscode\PatternAnalysis-2026\recognition\hipmri_3d_improved_unet
python -m pip install -r requirements.txt
python -m pytest -v
```

Before a final claim, run the data audit and full baseline/improved training on Rangpur, retain configuration/log/checkpoint/metric artifacts, and inspect 3-5 held-out cases. The tutor check-off, PR, Turnitin receipt and teaching-team review are separate external requirements.
