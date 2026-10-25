# HipMRI: Baseline and Improved 3D U-Net

COMP3710 · s4910970 · Report updated 26 October 2026

This project investigates whether residual volumetric features and deep supervision improve pelvic MRI segmentation enough to justify their resource cost. It segments Background, Body, Bones, Bladder, Rectum and Prostate from the CSIRO [Labelled weekly MR images of the male pelvis](https://doi.org/10.25919/45T8-P065) dataset, called **HipMRI** here as course shorthand. The intended use is a research contour-drafting assistant with human review, not autonomous treatment planning.

**Main finding.** In the original matched held-out experiment, Improved achieved patient-mean prostate Dice **0.813587**, against **0.811548** for Baseline. The gain was only **0.002038**, and the worst patient deteriorated. The evidence supports retaining Baseline as the conservative operational reference, rather than claiming a reliable clinical benefit from extra complexity.

## 1. Model and evaluation identity

The report's **Improved** is `LegacyImprovedUNet3D` (`legacy_improved_v1`), used in formal jobs **683675 / 683676**, code `4232e0d`. It is distinct from the later `improved_context_v2` and lightweight-decoder experiments. Their validation scores are not substituted for this model's test results.

Both models use channels 16/32/64/128, three max-pooling stages, transposed-convolution upsampling and concatenated encoder skips. Baseline uses two Conv3d–InstanceNorm3d–LeakyReLU operations per block. Improved adds residual projections and two training-only auxiliary heads. Its inference uses the final head; it does **not** use Context-v2 dropout or multi-scale logit fusion. Raw logits enter cross entropy; softmax is applied inside soft Dice. Auxiliary targets use nearest-neighbour downsampling.

![Baseline and historical Improved architecture](docs/assets/model_architecture.drawio.png)

The objective is weighted cross entropy plus weighted soft Dice, with prostate weight 4 in both terms. Improved adds each auxiliary objective at weight 0.25. The loss therefore changes with the architecture; this comparison measures the complete architecture-and-supervision package.

## 2. Feasibility review

**User need and scope.** A contouring specialist needs candidate masks that reduce editing without hiding local prostate omissions. The prototype investigates a 3D Baseline versus residual Improved under the same patient split. It makes no claim about time saved by clinicians, clinical safety or native-resolution treatment contours.

**Acceptance criteria.** (1) Zero patient overlap, with paired MRI and labels spatially aligned. (2) Held-out mean patient prostate Dice at least 0.70, with every foreground class reported separately. (3) A meaningful upgrade should gain at least 0.01 mean prostate Dice over Baseline, lose no more than 0.02 on the worst patient and avoid a material FN increase. (4) The engineering screen retains a maximum 1.35-times training-memory ratio. (5) At least three actual failure cases must be inspected, including missed tissue and excessive contour extent. Criteria 3–4 are project engineering targets, not thresholds stated by the course or evidence of clinical acceptability. Later validation screens use aggregate FN at most 1.10-times and full-validation time at most 2-times their declared control. Those fields were not all preserved in the original formal summary, so a complete retrospective pass is not claimed.

**Choice and preliminary evidence.** The audit found 211 volumes from 38 patients with labels 0–5. Patient-disjoint splitting and a small volumetric input establish feasibility before expensive runs. A downsampled 3D Improved U-Net investigates the course's Hard-tier residual/deep-supervision option; choosing that option does not establish marks or clinical utility. Three-dimensional kernels capture local cross-slice context, while residual paths and auxiliary objectives test an alternative optimization design [1,2]. Both report models are 3D, so this study cannot establish superiority of 3D over 2D.

**Budget, risks and fallback.** Training uses one A100, batch 1, 96-cubed crops and 20 epochs; full evaluation uses reconstructed 128-cubed volumes. Risks include patient leakage, changing label identities, loss of thin structures after resizing, class imbalance and architecture/checkpoint mismatch. Controls are a frozen manifest, cited label mapping, nearest-neighbour masks, prostate-weighted losses and strict version-aware loading. If Improved fails the upgrade criteria, retain the tested Baseline and report the trade-off. No failed criterion is relaxed after results are seen.

**Outcome and consultation status.** Data isolation and aggregate test Dice are supported. The formal mean gain was below 0.01 and the worst patient fell by 0.062061; the upgrade was not accepted. Subsequent validation searches also failed to produce a candidate satisfying all declared conditions. This section consolidates the review and observed outcome; it does not certify that the required midpoint tutor consultation was completed. That external record remains to be confirmed by the student.

## 3. Data and reproducible protocol

The frozen split contains **27/6/5 patients** and **146/46/19 volumes** for train/validation/test. All weekly scans of one patient stay together. [Manifest](evidence/dataset_manifest.json) SHA-256: `5457a611c5611f944a26d214a1f325b803dded66f70df5fda3fd009642751a07`. The [CSIRO label description](evidence/sources/csiro-brief-data-description.txt) documents `5=Prostate`; label identities are not inferred from anatomy or mask size.

Images are z-score normalized and resized with trilinear interpolation; integer masks use nearest neighbour. Volumes are transposed to D/H/W without an additional anatomical reorientation. Training samples a 96-cubed crop, with probability 0.5 of selecting a crop containing prostate. Both original formal arms used no augmentation, Adam at 0.001, batch 1, seed 42 and 20 epochs. The historical checkpoint rule was **highest validation all-class mean Dice**, selecting Baseline epoch 20 and Improved epoch 18. Patient-prostate validation selection was introduced only in later studies.

Evaluation uses the same five patients, 128-cubed target geometry, 128-cubed windows, stride 64 and overlap-averaged logits. These test patients have been evaluated by several historical models; they are not an untouched new external cohort. No later validation candidate has a new held-out result. Metrics describe the resized grid, not native-resolution geometric accuracy.

The headline averages prostate Dice across scans within each patient, then gives each of the five patients equal weight. The stored `per_class` Dice/IoU instead averages available scan-level class scores. Its Prostate Dice therefore differs from the patient-weighted headline. Background is excluded from the foreground table. This study reports no HD95 or physical-distance claims.

### Running the maintained implementation

Install `requirements.txt`; the [exact local verification versions](docs/evidence/requirements-local-verified.txt) are recorded separately. Formal runs recorded PyTorch **2.13.0+cu130**, CUDA **13.0** and an **A100-PCIE-40GB**. Their complete historical pip freeze was not retained; the local environment must not be presented as that missing GPU lockfile. A repeated seed is not a guarantee of bitwise replay.

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest==9.1.1
python -m pytest -q
python src/audit.py data/HipMRI_Study_open --output local/re_audited_manifest.json --seed 42
```

Obtain the MRI data separately; do not upload scans or weights to the course repository. Compare the regenerated manifest with the frozen one before training. A current-interface command corresponding to the historical training configuration is:

```powershell
python src/train.py --manifest evidence/dataset_manifest.json --data-root data/HipMRI_Study_open --output-dir artifacts/new_baseline --model baseline --epochs 20 --batch-size 1 --learning-rate 0.001 --base-channels 16 --target-shape 128 128 128 --train-patch-shape 96 96 96 --foreground-patch-probability 0.5 --seed 42 --device cuda --prostate-label 5 --prostate-label-source "CSIRO DOI 10.25919/45T8-P065, BriefDataDescription.txt: 5=Prostate" --prostate-ce-weight 4 --prostate-dice-weight 4 --selection-metric mean_dice --scheduler none
```

For the report's Improved, change the output directory and replace `--model baseline` with `--model improved --architecture-version legacy_improved_v1`. Do not enable augmentation for this historical comparison. These commands reproduce the declared configuration through the current interface, not an exact replay of the historical runtime; [original commands](docs/evidence/improved_original_commands.txt) are retained.

```powershell
python src/predict.py --checkpoint checkpoints/improved_legacy_best.pt --manifest evidence/dataset_manifest.json --data-root data/HipMRI_Study_open --output-dir artifacts/frozen_test --patch-size 128 128 128 --stride 64 64 64 --device cuda
```

The checkpoint path is an example; weights are retained locally, not bundled. Inference strictly loads the saved architecture. This evaluates the held-out partition: use it only for a frozen evaluation, never to tune a model. [Source and evidence map](docs/evidence/README.md) identifies retained checkpoints and metrics.

## 4. Original matched held-out results

Both formal jobs finished `COMPLETED|0:0`, confirmed by [final accounting](docs/evidence/sacct_final_20261026.txt). Baseline and Improved ran for 31:32 and 31:15 wall time respectively.

### Foreground class metrics

| Class    | Baseline Dice | Improved Dice | Baseline IoU | Improved IoU |
| -------- | ------------: | ------------: | -----------: | -----------: |
| Body     |      0.958148 |      0.977807 |     0.919909 |     0.956600 |
| Bones    |      0.859094 |      0.904555 |     0.753894 |     0.825973 |
| Bladder  |      0.957028 |      0.960610 |     0.917811 |     0.924386 |
| Rectum   |      0.852767 |      0.861629 |     0.747437 |     0.758714 |
| Prostate |      0.822599 |      0.831385 |     0.702348 |     0.715546 |

All five aggregate foreground Dice values exceed 0.70. This does not mean every patient or every boundary is acceptable.

### Patient-weighted prostate comparison

| Patient                     |           Baseline |           Improved | Improved − Baseline |
| --------------------------- | -----------------: | -----------------: | -------------------: |
| B037                        |           0.736256 |           0.674195 |            -0.062061 |
| H007                        |           0.781378 |           0.839755 |            +0.058377 |
| K042                        |           0.887015 |           0.878500 |            -0.008516 |
| L011                        |           0.781136 |           0.795553 |            +0.014417 |
| M013                        |           0.871957 |           0.879931 |            +0.007974 |
| **Mean (5 patients)** | **0.811548** | **0.813587** |  **+0.002038** |

Improved improved three patients and regressed on two. The worst patient, B037, fell from 0.736256 to 0.674195 despite the small rise in the mean. A single seed and five previously evaluated patients do not demonstrate statistical significance or stable generalization.

### Recorded computational cost

| Metric                                |  Baseline |  Improved |
| ------------------------------------- | --------: | --------: |
| Parameters                            | 1,400,646 | 1,423,106 |
| Peak training allocated memory (MiB)  |     977.6 |   1,193.4 |
| Peak inference allocated memory (MiB) |   1,047.4 |   1,047.5 |
| Training-memory ratio to Baseline     |     1.000 |     1.221 |
| Recorded forward time / volume (s)    |  0.060885 |  0.064924 |

Forward time is recorded total inference time divided by 19 scans; GPU timing is synchronized around reconstruction and excludes file reading/preprocessing. It was not a separate controlled warm-up benchmark. The saved `mean_patient_inference_seconds` is a mean of patient totals, not single-volume latency. Resource ratios alone do not compensate for missed tissue.

![Actual formal training histories](docs/assets/submission/formal_training.png)

The dots identify validation-selected checkpoints. The retained historical histories contain all-class validation Dice, not patient-prostate validation Dice; no missing curve has been fabricated.

## 5. Failure cases and boundary utility

Cases were selected by the retained deterministic [selection rule](docs/evidence/failure_selection.json): rank scans by mean Baseline/Improved FP+FN, with patient/case tie-breaks; display the highest-error slice within the ground-truth prostate extent. These are five scans from three patients, not five independent patients. Every panel compares the same MRI, reference and slice; **white = prostate FP, red = prostate FN**. Counts and Dice below cover the full resized volume, not the displayed slice.

| Scan       | Baseline volume Dice | Improved volume Dice | Baseline FN / FP | Improved FN / FP |
| ---------- | -------------------: | -------------------: | ---------------: | ---------------: |
| B037_Week0 |             0.736256 |             0.674195 |       2495 / 393 |       3123 / 166 |
| L011_Week2 |             0.748694 |             0.749511 |         68 / 894 |        159 / 738 |
| L011_Week7 |             0.742469 |             0.763189 |         81 / 868 |        147 / 661 |
| K042_Week0 |             0.887015 |             0.878500 |        439 / 432 |        673 / 208 |
| L011_Week0 |             0.794687 |             0.819822 |        109 / 803 |        170 / 579 |

**B037 Week0: missed prostate.** Improved reduces FP but increases FN from 2,495 to 3,123. The selected slice shows extensive missing prostate, consistent with under-contouring rather than a harmless boundary adjustment. Reject this scan for unattended drafting; the mean improvement elsewhere does not repair the failure.

![B037 Week0 matched errors](docs/assets/submission/failure_B037_Week0.png)

**L011 Week2: overextension remains.** FP falls from 894 to 738 while FN rises from 68 to 159; volume Dice barely changes. Inspect the contour extent and small residual errors rather than interpreting the lower FP as a complete success.

![L011 Week2 matched errors](docs/assets/submission/failure_L011_Week2.png)

**L011 Week7: partial improvement.** FP decreases by 207 voxels and Dice rises, but FN also rises from 81 to 147. This supports a trade-off between excess contour and missed tissue, not uniform boundary recovery.

![L011 Week7 matched errors](docs/assets/submission/failure_L011_Week7.png)

**K042 Week0: high overlap still hides regression.** Dice stays near 0.88, but FN increases from 439 to 673 as FP falls. This relatively high-scoring scan still needs local review.

![K042 Week0 matched errors](docs/assets/submission/failure_K042_Week0.png)

**L011 Week0: reduced FP with residual omissions.** FP falls from 803 to 579 and Dice improves; FN nevertheless rises from 109 to 170. This is a potential editing aid, subject to visual correction, rather than a safe autonomous contour.

![L011 Week0 matched errors](docs/assets/submission/failure_L011_Week0.png)

![B037 errors across slices](docs/assets/submission/B037_slice_errors.png)

The slice profile shows where omissions occur across the resampled prostate extent. Low/high depth support and the central region should all be inspected; the images do not provide clinician-confirmed base/apex annotations or a millimetre-distance evaluation. Thus this experiment does not establish that either model eliminates apex/base failures, rectal-wall leakage or the need for expert review.

## 6. Subsequent validation experiments and GIF

These are supplementary investigations under a later patient-prostate validation selection rule. They do not replace the original formal test comparison.

| Validation candidate             | Mean patient prostate Dice | Finding                                                                         |
| -------------------------------- | -------------------------: | ------------------------------------------------------------------------------- |
| Context v2 B (692354)            |                   0.829472 | Highest retained validation reference; training memory 1.486-times historical D |
| P1: lower LR + cosine, 20 epochs |                   0.802803 | No gain over B                                                                  |
| P2: fixed LR, 30 epochs          |                   0.825796 | Lower FN, but below B and memory limit failed                                   |
| P3: lower LR + cosine, 30 epochs |                   0.811962 | Below B; FN/memory gates failed                                                 |
| E: augmented Baseline            |                   0.791160 | No gain in this seed                                                            |
| Improved width14                 |                   0.770696 | Memory passed; accuracy/FN failed                                               |
| R0: fresh Context v2 control     |                   0.819498 | Same seed did not reproduce B numerically                                       |
| R1: compact decoder              |                   0.765992 | About 25.9% less training memory than R0; all six patients regressed            |
| R2: mild intensity augmentation  |                   0.795643 | Five of six patients regressed relative to R0                                   |

R1 jointly changed decoder convolutions, residual paths, normalization and dropout; its result cannot identify one component as the cause. R2 changed several intensity ranges. These are single-seed package comparisons, not stable effects. No candidate passed every declared engineering condition, and no subsequent held-out test was run. Retrospective small gains in a repeatedly inspected validation cohort should be treated cautiously.

![Context-v2 validation progression — supplementary only](docs/assets/context_v2_ablation_692354/B_progress.gif)

This GIF was generated from this project's fixed validation case and saved predictions. It illustrates Context v2 only; it is not a visualization of the historical tested Improved. The PDF uses a static GIF frame; the Markdown retains animation. Reference-project source organization was studied, but its scores, masks and animations are not this project's evidence.

## 7. Engineering recommendation and limitations

Retain the tested Baseline as the conservative reference and require human correction of draft contours. The historical Improved's +0.002038 patient-mean gain does not meet the project's +0.01 meaningful-upgrade target, and B037 is materially worse. It is reasonable to report an unsuccessful upgrade: the original Improved passes aggregate class overlap targets, but does not demonstrate sufficient operational benefit.

The evidence answers the narrower question of added residual/deep-supervised 3D complexity, not a 2D-versus-3D comparison. No clinician editing-time study, external-cohort validation, statistical-significance claim or physical boundary-distance result is available. Next investigations should resolve reproducibility and use paired training seeds before making a stability claim. Further testing must follow a frozen protocol, with prior exposure of these test patients disclosed. No acceptance threshold is changed to make a failed experiment pass.

## References

1. Çiçek et al. *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation* (2016). [Paper](https://arxiv.org/abs/1606.06650).
2. Isensee et al. *Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge* (2018). [Paper](https://arxiv.org/abs/1802.10508). Architectural inspiration does not establish direct reproduction.
3. CSIRO. *Labelled weekly MR images of the male pelvis*, DOI [10.25919/45T8-P065](https://doi.org/10.25919/45T8-P065); `BriefDataDescription.txt`, Labels block.

## Artificial Intelligence Usage Disclosure
