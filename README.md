# HipMRI 3D Improved U Net for Prostate Segmentation

## Scope

This is the COMP3710 Hard segmentation project. It uses downsampled HipMRI 3D NIfTI volumes from `/home/groups/comp3710/HipMRI_Study_open` to test whether volumetric residual modelling and deep supervision reduce clinically important prostate boundary and cross-slice errors enough to justify their compute cost.

The implementation compares a plain 3D U-Net with a 3D Improved U-Net. The improved model contains volumetric residual units and two decoder deep-supervision heads. This is distinct from the repository's OASIS 2D work and must not reuse its reported values.

## Data and validity controls

`src/dataset.py` pairs `semantic_MRs` and `semantic_labels_only` by their relative NIfTI paths. It validates 3D shape, affine agreement, finite intensity values and integral labels. Splits are by patient, not slice or patch, and the exact seed-42 manifest must be reused by both models.

No HipMRI data, checkpoints or final results are committed. The code has been checked against synthetic NIfTI volumes locally; real performance, memory use and image orientation must be audited on Rangpur before reporting results.

The semantic-mask mapping is now frozen in [`evidence/SOURCES.md`](evidence/SOURCES.md): the authoritative CSIRO data description (collection v2, DOI `10.25919/45T8-P065`, `Labels:` block) states `5=Prostate`. Formal runs must pass this exact mapping and citation into the checkpoint configuration; it is not inferred from mask statistics.

### Local Windows data root

For local audit and development, the ignored raw 3D dataset is stored at `E:\vscode\PatternAnalysis-2026\data\HipMRI_Study_open`. Its direct children must be `semantic_MRs` and `semantic_labels_only`; do not rename individual NIfTI files or commit this directory. From the repository root, run:

```powershell
$dataRoot = Resolve-Path ".\data\HipMRI_Study_open"
python src\audit.py $dataRoot --output evidence\dataset_manifest.json --seed 42
```

`src/audit.py` receives the data root as an argument rather than embedding a local path in Python. On Rangpur, pass `/home/groups/comp3710/HipMRI_Study_open` to the same command. This preserves one code path across local audit and formal GPU training.

## Confirmed evaluation protocol (staff reply, 2026-10-21)

The plain 3D U-Net is the baseline. Both models must use the same patient-level split, preprocessing, downsampling and evaluation protocol. The 211 volumes come from only 38 patients according to the staff reply, so neither volumes, slices nor patches may be split independently across train, validation and test.

The headline result is **prostate Dice**. For each held-out test patient, reconstruct the complete volume from its sliding-window logits, convert the reconstructed logits to the final segmentation, and compute Dice for the prostate label. Report the arithmetic mean of those patient-level prostate Dice values. The `>= 0.70` target applies to this value; it is not a pooled-voxel Dice, foreground macro Dice or a mean over every label.

Report Dice for every non-background semantic label as supplementary class-wise results. Do not report background Dice. A formal run must record its patch size, overlap, logit-merging method and the fact that metrics were computed only after full-volume reconstruction. Comparative panels use a predeclared error-relevant slice inside the ground-truth prostate extent; they do not make an unverified anatomical “prostate base” claim.

The staff reply does **not** identify which of the observed labels `0`–`5` is prostate. The authoritative mapping is separately recorded in [`evidence/SOURCES.md`](evidence/SOURCES.md): `5=Prostate`. Record that citation in the immutable experiment configuration and use the same mapping for both models. Do not infer any label from label frequency, mask size or label numbering.

## Feasibility review

**Need and scope.** The intended user is a radiotherapy contouring specialist. The prototype drafts a 3D segmentation for review; it is not an autonomous clinical decision system.

**Acceptance criteria.** Zero patient overlap; identical test patients for both models; mean per-held-out-patient prostate Dice at least 0.70 after full-volume reconstruction; Dice/IoU for every non-background label, peak VRAM and per-case inference time; 3-5 real failure cases selected by a fixed common-error rule.

**Smoke evidence, not a formal result.** Rangpur A100 40 GB jobs `680366` (baseline) and `680381` (improved) each completed one epoch, checkpoint selection and held-out full-volume inference with the same manifest. Their patient-level prostate Dice values were both `0.0`; this demonstrates the real-data execution chain only and must not be used as a model comparison or target claim. Formal runs record `sacct`, peak VRAM, per-patient inference time and prostate-specific comparison panels.

**Local sampling audit, not model evidence.** `artifacts/patch_audit_B006_Week0_128_96.json` records 200 deterministic samples for each foreground probability on `B006_Week0`, with target geometry `128³` and training patch geometry `96³`. For this case, both `p=0.50` and `p=0.75` yielded prostate-containing patches in all 200 samples and the same mean prostate count (`2552` voxels). The large patch geometry already covers prostate for every sampled crop in this case; this is sampling-path evidence only, not a Dice or performance result.

## Formal matched experiment results

Rangpur formal jobs `683675` (Baseline) and `683676` (Improved) both completed normally (`COMPLETED|0:0`). They used commit `4232e0d`, manifest SHA-256 `5457a611c5611f944a26d214a1f325b803dded66f70df5fda3fd009642751a07`, seed 42, the same 27/6/5 patient split, target geometry `128³`, training patches `96³`, foreground-patch probability `0.50`, prostate CE/Dice weights `4/4`, batch size 1, base channels 16, learning rate `0.001`, and full-volume overlap-averaged-logit inference with patch `128³` and stride `64³`. The only recorded experiment-configuration difference is `model`.

`best.pt` was selected on validation data only: epoch 20 for Baseline and epoch 18 for Improved. The held-out test set was evaluated once after that selection; no reported test patient was used to change a loss weight, sampling probability, model component, epoch budget or inference setting.

| Metric | Baseline | Improved | Difference (Improved - Baseline) |
| --- | ---: | ---: | ---: |
| Mean patient-level prostate Dice (headline; 5 held-out patients) | 0.8115 | 0.8136 | +0.0020 |
| Mean patient inference time (s) | 0.2314 | 0.2467 | +0.0154 |
| Peak inference VRAM (MB) | 1047.4 | 1047.5 | +0.1 |
| Peak training VRAM (MB) | 977.6 | 1193.4 | +215.7 |
| Training time (s) | 1434.4 | 1417.6 | -16.9 |
| Parameter count | 1,400,646 | 1,423,106 | +22,460 |

Both models exceed the predeclared held-out mean patient-level prostate Dice target (`>= 0.70`). The mean gain of `0.0020` is not consistent across the five held-out patients: B037 changes by `-0.0621`, H007 by `+0.0584`, K042 by `-0.0085`, L011 by `+0.0144`, and M013 by `+0.0080`. The exploratory two-sided 95% t interval for the paired mean difference is `[-0.0520, 0.0561]` (`n=5`, `df=4`), so it includes both clinically relevant negative and positive directions. This experiment does not establish a statistically significant or practically clear prostate-segmentation advantage for Improved.

| Non-background label | Baseline Dice / IoU | Improved Dice / IoU |
| --- | ---: | ---: |
| 1 | 0.9581 / 0.9199 | 0.9778 / 0.9566 |
| 2 | 0.8591 / 0.7539 | 0.9046 / 0.8260 |
| 3 | 0.9570 / 0.9178 | 0.9606 / 0.9244 |
| 4 | 0.8528 / 0.7474 | 0.8616 / 0.7587 |
| 5 (Prostate) | 0.8226 / 0.7023 | 0.8314 / 0.7155 |

The class-wise rows aggregate label voxels and are supplementary. The headline above remains the arithmetic mean of complete-volume, patient-level prostate Dice.

### Paired prostate-only error inspection

`src/compare_failures.py` compared the two models only after verifying matching manifest, prostate label/source, geometry, patch, stride, test-patient set and prediction-case set. Its selection manifest records commit `4232e0d91d8aa6cf8c1de38883b213c07cc62d10` and the fixed rule: mean FP+FN across both models, patient/case tie-break, then the highest-error slice inside the ground-truth prostate extent. The five generated panels are `B037_Week0`, `L011_Week2`, `L011_Week7`, `K042_Week0` and `L011_Week0`; they are error-selected prostate-containing slices, not anatomical prostate-base slices.

B037 is retained rather than excluded: Improved has fewer false positives on its selected slice (166 vs 393) but substantially more false negatives (3123 vs 2495), consistent with its lower patient-level prostate Dice. Across the selected cases, Improved frequently trades false positives for false negatives rather than showing a uniformly better prostate contour. For compute cost, training VRAM and worst-case patient robustness, this evidence supports Baseline as the engineering recommendation. Improved remains a research candidate for investigating supplementary multi-class metrics, not the default recommended model.

The evidence archive contains the final Slurm accounting record, the comparison selection manifest and five PNG panels; it is kept outside Git because it contains derived medical-image material. No raw data, checkpoints or prediction tensors are committed.

**Risks, budget and fallback.** 3D memory demand, small foreground regions, NIfTI orientation, slow I/O and failed cluster jobs require a short GPU smoke run before full Slurm training. If the improved model does not improve real held-out boundary errors at acceptable cost, report the plain 3D U-Net as the recommended fallback.

## Current verification

```powershell
cd E:\vscode\PatternAnalysis-2026
python -m pip install -r requirements.txt
python -m pytest -v
```

The completed formal run retains configuration, log, checkpoint, metric and resource artifacts, together with five paired held-out error panels. Future work may use new patient-disjoint splits or multiple seeds, and may investigate B037-like over- or under-segmentation on validation data. It must not use the current held-out patients to tune this completed comparison. The tutor check-off, PR, Turnitin receipt and teaching-team review remain separate external requirements.
