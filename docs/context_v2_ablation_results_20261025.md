# Context v2 A/B/C/D validation ablation — 25 October 2026

**B (Improved Context v2 with training augmentation and fixed learning rate) ranks first on validation mean patient prostate Dice, but fails the predeclared training-memory gate. No arm is eligible for test review under the current rules, and this experiment supplies no new held-out test conclusion.**

## Evidence and matched protocol

This record describes Rangpur job `692354`. The retained [generated validation report](../artifacts/2026-10-25/692354_context_v2_ablation/context_v2_ablation_692354/runs/validation_report.md) and [structured comparison](../artifacts/2026-10-25/692354_context_v2_ablation/context_v2_ablation_692354/runs/validation_comparison.json) are the numerical sources. Raw artifacts remain in the local, Git-ignored artifact tree; these relative links require the retained package and do not distribute checkpoints.

| Arm | Model | Training augmentation | Learning-rate schedule | Controlled contrast |
| --- | --- | --- | --- | --- |
| A | Improved Context v2 | Off | Fixed 0.001 | Architecture versus D |
| B | Improved Context v2 | On | Fixed 0.001 | Augmentation versus A |
| C | Improved Context v2 | On | Cosine, minimum 0.00001 | Scheduler versus B |
| D | Baseline | Off | Fixed 0.001 | Matched baseline |

All arms share manifest SHA-256 `5457a611c5611f944a26d214a1f325b803dded66f70df5fda3fd009642751a07`, seed 42, 20 epochs, 16 base channels, batch size 1, `128 x 128 x 128` target, `96 x 96 x 96` training crop, foreground sampling probability 0.5, prostate label 5, prostate CE/Dice weights 4/4, and validation mean patient prostate Dice checkpoint selection. Validation covers six patients. Augmentation is disabled for validation/test. The contrasts isolate the declared configuration changes within this split and seed; A versus D compares an architecture package, not individual architectural components.

## Validation results and resources

The metric below is the arithmetic mean of patient prostate Dice, not the all-class `validation_mean_dice` field. FN and FP are aggregate validation prostate voxel counts. Validation seconds refer to the selected epoch's full validation pass, not per-patient test inference latency.

| Arm | Selected epoch | Mean patient prostate Dice | Worst patient Dice | FN | FP | Parameters | Peak training MiB | Total training seconds | Validation seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | 19 | 0.829472 | 0.778456 | 13,159 | 17,733 | 1,628,996 | 1,455.8 | 1,441.3 | 21.19 |
| A | 19 | 0.818156 | 0.772367 | 11,227 | 21,313 | 1,628,996 | 1,455.8 | 1,442.8 | 21.26 |
| D | 17 | 0.810463 | 0.722912 | 13,804 | 19,624 | 1,400,646 | 979.6 | 1,386.9 | 21.46 |
| C | 13 | 0.785744 | 0.752366 | 16,720 | 25,065 | 1,628,996 | 1,455.8 | 1,442.6 | 21.16 |

| Contrast | Mean Dice difference | Worst-patient Dice difference | Interpretation within this experiment |
| --- | --- | --- | --- |
| A minus D | +0.007693 | +0.049456 | Architecture package improves observed validation scores, but misses the +0.01 mean-gain gate |
| B minus A | +0.011316 | +0.006088 | Augmentation is associated with the main additional mean-Dice gain |
| C minus B | -0.043727 | -0.026090 | This cosine configuration underperforms fixed LR within the 20-epoch budget |
| B minus D | +0.019009 | +0.055544 | Highest observed mean and cohort minimum, subject to the failed memory gate |

B improves mean Dice by approximately 2.35% relative to D and the cohort minimum by 7.68%. Aggregate FN falls by 645 (4.67%) and FP by 1,891 (9.64%). Against A, B increases FN by 1,932 but reduces FP by 3,580: total FN plus FP falls by 1,648. These voxel totals help explain the error trade-off; they are not interchangeable with the patient-averaged Dice metric.

The cohort minimum is not a matched-patient improvement: D's worst patient is H017, whereas B's is S022. B improves four of six patient Dice scores versus D; M030 and S022 decline. Therefore, neither universal patient benefit nor reduced variance is established. One seed and one small validation split do not establish statistical significance or generalization.

C has higher FN and FP than B and lower mean and minimum Dice. Retain it as a negative ablation result and do not promote this configuration. The result does not establish that cosine schedules generally fail; alternative schedules or longer budgets require a separately specified experiment.

## Resource screening and decision

The predeclared engineering rules require mean Dice gain at least 0.01, worst-patient drop no greater than 0.02, FN ratio no greater than 1.10, peak training-memory ratio no greater than 1.35, and validation-time ratio no greater than 2.0, all against D. These are project screening criteria, not course requirements, statistical tests or clinical assurance.

| Arm | Mean gain | Worst patient | FN ratio | Training memory | Validation time | Eligible for test review |
| --- | --- | --- | --- | --- | --- | --- |
| A | Fail | Pass | Pass | Fail | Pass | False |
| B | Pass | Pass | Pass | Fail | Pass | False |
| C | Fail | Pass | Fail | Fail | Pass | False |

B uses 1,455.8 MiB versus D's 979.6 MiB: **1.486 times baseline (+48.6%)**, exceeding the 1.35 limit. Its parameter count is 16.3% higher and total training duration approximately 3.9% longer; selected-epoch validation time is similar (21.19 versus 21.46 seconds). Accuracy and memory must both pass; higher validation Dice does not override the failed resource criterion. The JSON records `eligible_for_test_review=false` and `automatic_test_permitted=false` for A, B and C.

Record B as the validation-first candidate requiring explicit resource review, retain A as the architecture control, retain C as a negative scheduler result, and retain D as the lower-memory comparator. Any later revision of the memory budget must be documented as a post-result policy change, with its rationale; it cannot be represented as passing the original gate. No new held-out evaluation is authorized by this documentation update.

Before any separately authorized test, review validation failure panels, freeze the checkpoint and complete configuration, and preserve the evaluation protocol. B's selected checkpoint is epoch 19 with SHA-256 `477244b5009d1b941af9f2e94ed8dbc67ac369771227faafd780eeb7fceb46fc`. Do not switch arms or epochs after inspecting test scores. The existing held-out cohort has already been viewed in historical experiments; a later reuse would not be an independent blind test.

## Artifact and accounting limits

The synchronized package contains 54 files, including four checkpoints, complete histories, configuration and learning-rate records, curves, epoch-1/epoch-5/best-epoch validation GIFs, manifest and environment records. No new `test_metrics.json`, test predictions or test failure panels were found in this package. The GIFs do not establish that the full validation failure-panel review has been completed.

The user reports final Rangpur `COMPLETED` status. The synchronized package does not contain an independently retained `sacct_final.txt`; archive the final `sacct -X` job-state and exit-code output for accounting evidence. Local histories and summaries support the numerical record but do not independently verify scheduler accounting. Historical formal and supplementary test conclusions remain unchanged.
