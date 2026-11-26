# PR #68 completeness audit

Checked on 26 November 2026 against the current PR head, `6bc9a99a781aa3333cb3d0adb6ed9e36dcb33025`.

## Result

All **19 of 19 source files** are present in [the submitted-file snapshot](original-pr68/), with **zero missing files and zero byte differences**. The source file set, on-disk bytes and committed Git blob IDs were checked independently. The [manifest](../evidence/pr68-manifest.json) records the original blob IDs.

The working repository already had identical copies of **15** source files outside the snapshot. The remaining **4** original variants are now preserved alongside the working versions: `.gitignore`, `README.md`, `assets/failure_case_k042_week0.png`, `compare_failures.py`. Existing files are retained.

| Original PR file | Working repository comparison | Submitted-file snapshot |
| --- | --- | --- |
| `.gitignore` | Original variant preserved in snapshot | Exact match |
| `README.md` | Original variant preserved in snapshot | Exact match |
| `assets/HipMRI_Feasibility_Review_v5.pdf` | Existing identical copy: `Submission-final_v2/assets/HipMRI_Feasibility_Review_v5.pdf` | Exact match |
| `assets/failure_case_b037_week0.png` | Existing identical copy: `Submission-final_v2/assets/failure_case_b037_week0.png` | Exact match |
| `assets/failure_case_h007_week0.png` | Existing identical copy: `Submission-final_v2/assets/failure_case_h007_week0.png` | Exact match |
| `assets/failure_case_k042_week0.png` | Original variant preserved in snapshot | Exact match |
| `assets/failure_case_l011_week2.png` | Existing identical copy: `Submission-final_v2/assets/failure_case_l011_week2.png` | Exact match |
| `assets/failure_case_m013_week0.png` | Existing identical copy: `Submission-final_v2/assets/failure_case_m013_week0.png` | Exact match |
| `assets/improved_h007_week0.gif` | Existing identical copy: `Submission-final_v2/assets/improved_h007_week0.gif` | Exact match |
| `assets/model_architecture.png` | Existing identical copy: `Submission-final_v2/assets/model_architecture.png` | Exact match |
| `assets/training_curves.png` | Existing identical copy: `Submission-final_v2/assets/training_curves.png` | Exact match |
| `audit.py` | Existing identical copy: `Submission-final_v2/audit.py` | Exact match |
| `compare_failures.py` | Original variant preserved in snapshot | Exact match |
| `dataset.py` | Existing identical copy: `Submission-final_v2/dataset.py` | Exact match |
| `modules.py` | Existing identical copy: `Submission-final_v2/modules.py` | Exact match |
| `plot_training.py` | Existing identical copy: `Submission-final_v2/plot_training.py` | Exact match |
| `predict.py` | Existing identical copy: `Submission-final_v2/predict.py` | Exact match |
| `requirements.txt` | Existing identical copy: `Submission-final_v2/requirements.txt` | Exact match |
| `train.py` | Existing identical copy: `Submission-final_v2/train.py` | Exact match |

## Completeness boundary

This verifies all files tracked in the PR project directory, including hidden `.gitignore`, Python source, dependencies, README, PDF, PNG figures and GIF. It does not claim to copy files that were never committed, such as raw MRI data, trained checkpoints or cluster outputs. The personal repository keeps its existing history; the original PR's 22 commits and reviews remain accessible through the PR links.
