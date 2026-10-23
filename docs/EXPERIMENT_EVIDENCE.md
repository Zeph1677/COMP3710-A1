# Experiment Evidence Index

This index distinguishes retained evidence from interpretation. Paths under `artifacts/` are intentionally Git-ignored; they remain local audit inputs and are not submission files. No raw HipMRI data, checkpoints, prediction tensors, archives, or Slurm logs are staged by this repository.

## Frozen data and label identity

| Evidence | Purpose | Identity or finding | Completeness |
| --- | --- | --- | --- |
| `evidence/dataset_manifest.json` | Patient-disjoint split and geometry audit | SHA-256 `188a42ad99e4a1a29ba6835828e3cc00a88edab9d691f7db753cfd0351e6b08a`; 211 volumes, 38 patients, split seed 42 | Present locally |
| `evidence/SOURCES.md` and `evidence/sources/csiro-brief-data-description.txt` | Semantic label interpretation | CSIRO v2 DOI `10.25919/45T8-P065`, `Labels:` block: `5=Prostate`; source SHA-256 `dc3a99204684ced859872d79a75707af3c0ad31d00f5f3213063ce5e30727ee6` | Present locally |
| `src/dataset.py`, `src/audit.py` | Pairing, split and preprocessing implementation | Relative PID-first pairing, affine/shape checks, patient-level split | Source tracked and tested |

## Main matched formal experiment: `4232e0d`

The primary conclusion uses jobs 683675 and 683676 only. Both share manifest hash, seed 42, target shape `128x128x128`, training patch `96x96x96`, foreground-patch probability 0.50, CE/Dice prostate weights 4/4, patch/stride `128^3/64^3`, overlap-averaged logits, and five held-out test patients. The historical selection field was `validation_mean_dice`; it is not interchangeable with the later prostate-specific selection rule.

| Local artifact path | Purpose | Recorded identity / result | Completeness |
| --- | --- | --- | --- |
| `artifacts/2026-10-22/683675_formal_baseline/{experiment_config.json,training_summary.json,test_metrics.json,git_sha.txt}` | Baseline train, selection and test | Git `4232e0d91d8aa6cf8c1de38883b213c07cc62d10`; selected epoch 20; prostate Dice 0.811548; 1,400,646 parameters | Downloaded locally |
| `artifacts/2026-10-22/683676_formal_improved/{experiment_config.json,training_summary.json,test_metrics.json,git_sha.txt}` | Improved train, selection and test | Same Git identity; selected epoch 18; prostate Dice 0.813587; 1,423,106 parameters | Downloaded locally |
| `artifacts/archives/rangpur_formal_results_20261022_4232e0d.zip` | Full retained formal-results archive | SHA-256 `197477e24118118f746b81269350f0e3be6ececcd34e1f8ccd4fb87ecea29f8e` | Archive present locally |
| `artifacts/archives/683675_683676_formal_full.tar.gz` | Source, histories, environments, scheduler records and predictions | Final rows: 683675 `COMPLETED|0:0`; 683676 `COMPLETED|0:0` | Archive present locally |
| `artifacts/archives/683675_683676_formal_paired_comparison.tar.gz` | Deterministic paired failure analysis | SHA-256 `d960896dd565c10c273c1e67534716e330b38892984cd1c855533fa481f49320`; selection manifest identifies B037_Week0 slice 70 | Archive present locally |

The test JSON files retain class-wise Dice/IoU, per-patient Dice, inference time, peak inference VRAM, patch geometry, overlap and logit merging method. The paired archive includes the saved predictions used for failure ranking; no new inference was run to create README figures.

## Validation scan and Wide20 supplementary comparison

Five Improved-U-Net validation candidates are retained at `artifacts/2026-10-22/validation_sweep/candidate_01_lr_low` through `candidate_05_wide20`. Wide20 was selected by validation patient-level prostate Dice, then recovered from checkpoint SHA-256 `6cbbcbccb0710643357bef0857f81ebc62b81ce664791f182b5af92253020f59`. This scan must not be represented as a blind held-out comparison.

| Local artifact path | Purpose | Recorded identity / result | Completeness |
| --- | --- | --- | --- |
| `artifacts/2026-10-23/686695_wide20_formal_control/` | Validation-only Wide20 control | Selected epoch 19; validation patient prostate Dice 0.797470; no test artifacts | Downloaded locally |
| `artifacts/2026-10-23/687753_formal_baseline_new/` | Matched Baseline-new arm | Checkpoint SHA-256 `370a44cf229df9e4af1bad2be8cbc68b1a73c60bb0a7651350248bb030731488`; test prostate Dice 0.818316 | Downloaded locally |
| `artifacts/2026-10-23/687754_formal_improved_wide20_recovery/` | Recovered Wide20 arm | Checkpoint SHA-256 `6cbbcbccb0710643357bef0857f81ebc62b81ce664791f182b5af92253020f59`; test prostate Dice 0.794166 | Downloaded locally |
| Rangpur read-only command on 2026-10-23 | Final scheduler accounting | `/opt/slurm/bin/sacct -X -j 687753,687754 --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocTRES -P -n` returned `COMPLETED|0:0` for both jobs | Live query completed |

The supplementary patient-level prostate Dice difference is `0.7941664101589291 - 0.8183164841240353 = -0.0241500739651062`. The result is a negative finding for this Wide20 configuration. It does not justify another selection cycle or a replacement of the main `4232e0d` comparison.

## Reproduction boundary

`src/train.py` records the validation history and validation-selected checkpoint. `src/predict.py` reads a checkpoint, verifies its manifest hash and cited prostate mapping, reconstructs whole test volumes with averaged overlapping logits, and writes the final test metrics. `src/compare_failures.py` rejects mismatched experiment fields before selecting common high-error cases. The commands and exact artifact-to-claim links are in the root `README.md`; environment distinctions are in `docs/EXPERIMENT_ENVIRONMENT.md`.
