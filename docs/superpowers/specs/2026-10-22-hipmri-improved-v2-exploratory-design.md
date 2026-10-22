# HipMRI Improved-v2 Exploratory Design

## Purpose and status

This design adds an exploratory `Improved-v2` architecture to the active HipMRI 3D project. It does not alter, replace, tag, archive, or rerun the completed Baseline formal experiment. The completed Baseline/Improved comparison at commit `4232e0d` remains the existing formal result. Improved-v2 is an exploratory follow-up because its design is informed by already inspected held-out results.

The objective is to test whether attention-gated skip connections reduce irrelevant high-resolution feature transfer while retaining the existing residual blocks and decoder deep supervision. The first evaluation boundary is validation only. The previously inspected five-patient held-out split must not be used to select V2 architecture, loss weights, epochs, patch settings, or checkpoints.

## Scope

Included:

- Add `AttentionGate3D` to `src/modules.py`.
- Add an explicit `ImprovedV2UNet3D` architecture.
- Use gates at the `dec3` and `dec2` encoder-to-decoder skip connections.
- Support `improved-v2` in training and checkpoint restoration.
- Compute patient-level prostate Dice over validation patients after complete-volume reconstruction.
- Select `best.pt` by validation patient-level prostate Dice, with deterministic secondary ordering.
- Keep the formal run's data manifest, seed, target shape, train patch, foreground sampling probability, loss weights, base channels, learning rate, batch size, epoch budget, and inference protocol unchanged for V2 preflight and validation experiments.
- Add local tests for gate shape/range/gradient flow, model restoration, validation selection, and output metadata.
- Label all V2 outputs and README text as exploratory.

Excluded:

- No Baseline-v2 retraining.
- No change to the original `BaselineUNet3D` or `ImprovedUNet3D` state dictionaries.
- No change to current formal result files, their README table, or their interpretation.
- No V2 selection based on the already inspected held-out test patients.
- No changes to base channels, optimizer, learning rate, sampling probability, prostate CE/Dice weights, target geometry, train patch, inference patch/stride, or auxiliary-loss weight in the first V2 study.
- No new formal Slurm submission until local tests and B006 preflight pass.

## Architecture

`AttentionGate3D` accepts an encoder skip tensor `x` and a decoder gating tensor `g`. It projects both tensors to a shared intermediate channel count, aligns `g` to the spatial size of `x`, computes a one-channel sigmoid coefficient map, and returns `x * alpha`. Its output shape exactly matches `x`; coefficients lie in `[0, 1]`.

`ImprovedV2UNet3D` uses the same residual block type, channel schedule, bottleneck, decoder blocks, final head, and two deep-supervision heads as `ImprovedUNet3D`. It differs only at the two high-level skip paths:

```text
up3(bottleneck) + gate(e3, up3(bottleneck)) -> dec3
up2(dec3)       + gate(e2, up2(dec3))       -> dec2
up1(dec2)       + direct e1 skip            -> dec1
```

The lower-resolution `e3` and `e2` paths are gated because they carry high-level anatomical context and directly affect the decoder stages where irrelevant activations can propagate. The highest-resolution `e1` path remains direct in V2 to limit added parameters and isolate the first architectural change.

## Training and checkpoint contract

The model identifier is `improved-v2`. It is stored in `experiment_config.json`, `best.pt`, `training_summary.json`, preflight summaries, command records, and output directory names. Checkpoint loading maps `baseline`, `improved`, and `improved-v2` to their exact architecture classes and rejects any other identifier.

Validation scoring uses the same prostate-label mapping recorded in the experiment config. For each validation patient, predictions are reconstructed at target geometry before prostate Dice is calculated; the selection score is the arithmetic mean of those patient-level scores. The sort key is:

1. Higher validation mean patient-level prostate Dice.
2. Higher worst validation patient-level prostate Dice.
3. Lower validation prostate FP+FN voxel count.
4. Earlier epoch, preserving deterministic selection when all metrics tie.

The previous all-class validation Dice remains a diagnostic history field but is not the V2 checkpoint-selection score. Every epoch history entry records the patient-level prostate mean, worst-patient Dice, prostate FP+FN count, validation runtime, and peak validation VRAM. The saved checkpoint and summary record the selection metric and selected values.

## Data boundaries and execution order

1. Local tests validate the new architecture and selection contract with synthetic data.
2. B006 train-only preflight runs 200 optimizer updates, using the existing `128^3` target geometry, `96^3` training patch, `p=0.50`, `4/4` prostate weights, seed 42, and the train partition only. The required minimum is prostate Dice `>= 0.70`.
3. A validation-only V2 run uses train and validation partitions for fitting and checkpoint selection. It must not invoke `predict.py` against the manifest test partition.
4. Results are reported as validation-selected exploratory evidence. Any later evaluation on the existing test split is explicitly post-hoc and cannot replace the completed formal conclusion.

The V2 run directories must be new and non-empty-directory protection remains active. Raw data, checkpoints, predictions, and derived medical images remain ignored by Git.

## Error handling and tests

`AttentionGate3D` rejects incompatible batch or channel layouts through normal tensor operations and explicitly aligns gating spatial dimensions before addition. Model construction rejects unsupported model identifiers. Checkpoint loading rejects a model identifier inconsistent with the saved architecture.

Tests cover:

- Attention output shape equals skip shape, coefficient range is bounded by sigmoid, and gradients reach skip/gating inputs and gate parameters.
- Improved-v2 preserves the existing training/evaluation return contract and has correct output sizes for divisibility-by-eight inputs.
- Training parser and checkpoint restoration recognise `improved-v2`.
- A controlled validation example selects the epoch with better patient-level prostate Dice even when all-class validation Dice is lower.
- Ties use worst-patient Dice, then lower prostate FP+FN, then earlier epoch.
- Preflight remains train-only and writes `improved-v2` metadata.

## Acceptance criteria

Local acceptance requires the full test suite, `bash -n` for both Slurm scripts, and `git diff --check` to pass. B006 preflight must produce a complete `preflight_summary.json`, exactly 200 optimizer updates, train-only provenance, and prostate Dice at least `0.70`.

Validation acceptance requires a complete V2 config/history/summary/checkpoint set, explicit patient-level prostate selection metadata, no test prediction directory, and recorded VRAM/runtime. No claim of a new unbiased held-out improvement is permitted from this design.
