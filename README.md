# HipMRI 3D U-Net Segmentation

COMP3710 project for six-class pelvic MRI segmentation. Implements a Baseline 3D U-Net and Improved variants with residual context blocks, dropout and multi-scale predictions.

## Core files

| File | Purpose |
| --- | --- |
| `src/audit.py` | Pair MRI/mask files and generate a patient-disjoint split manifest |
| `src/dataset.py` | Loading, preprocessing and train-only augmentation |
| `src/modules.py` | Baseline, Improved and version-aware checkpoint architecture |
| `src/train.py` | Training, validation and checkpoint selection |
| `src/predict.py` | Checkpoint inference and full-volume evaluation |
| `src/plot_training.py` | Training curves |
| `src/validation_visuals.py` | Optional validation GIFs used by training |

## Setup and training

```powershell
python -m pip install -r requirements.txt
python src/audit.py --help
python src/train.py --help
python src/predict.py --help
```

Obtain the course [CSIRO dataset](https://doi.org/10.25919/45T8-P065) separately and place `semantic_MRs/` and `semantic_labels_only/` under `data/HipMRI_Study_open/`. Data, generated manifests and checkpoints are excluded from Git.

```powershell
python src/audit.py data/HipMRI_Study_open --output evidence/dataset_manifest.json --seed 42
python src/train.py --manifest evidence/dataset_manifest.json --data-root data/HipMRI_Study_open --output-dir artifacts/baseline --model baseline --epochs 20 --batch-size 1 --base-channels 16 --target-shape 128 128 128 --train-patch-shape 96 96 96 --foreground-patch-probability 0.5 --selection-metric prostate_dice --prostate-label 5 --prostate-label-source "CSIRO HipMRI course label documentation: 5=Prostate" --prostate-ce-weight 4 --prostate-dice-weight 4
```

Confirm the class mapping against the dataset documentation before training. Labels are `0=Background`, `1=Body`, `2=Bones`, `3=Bladder`, `4=Rectum`, `5=Prostate`. Keep every patient's weekly scans in the same partition. The retained seed-42 split contains 27 training, 6 validation and 5 held-out patients.

For the documented Improved context architecture, use a fresh output directory and add `--model improved --architecture-version improved_context_v2 --augmentation` in place of `--model baseline`. Checkpoint architecture versions must match at inference. Held-out evaluation is a separate step and requires an approved frozen protocol.

## Evidence boundary

Experiment 692354 recorded validation mean patient prostate Dice of 0.810463 (Baseline D) and 0.829472 (Improved B). Improved used 1.486 times the peak training memory, exceeding the predeclared 1.35 limit. These are historical validation results, not held-out performance or results produced by the current uncommitted code.

This repository intentionally publishes only core source, runtime dependencies, license and this README. Local plans, configurations, tests, experiment tools, reports, logs, images, datasets and model weights are excluded. Historical evidence is retained locally and is not distributed with this minimal source repository.

## References

- Çiçek et al. *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation* (2016), arXiv:1606.06650.
- Isensee et al. *Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge* (2018), arXiv:1802.10508.

## Artificial Intelligence Usage Disclosure
