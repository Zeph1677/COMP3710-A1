# Local cleanup and submission status (2026-10-23)

## Changes verified locally

- Fixed shared semantic-label colors and discrete FP/FN colors in comparative figures.
- Re-rendered five historical selected cases from saved predictions, without training or inference.
- Preserved the source experiment SHA and original selection metadata in `assets/failure_selection.json`.
- Normalized the manifest to LF, matching checkpoint SHA-256 `5457a611c5611f944a26d214a1f325b803dded66f70df5fda3fd009642751a07`; the JSON content is unchanged.
- Corrected preprocessing order, example loss weights, and AI disclosure in the README.
- Removed unused generic course templates `README_COURSE_TEMPLATE.md` and `RECOGNITION.md`.
- Removed six internal planning/specification documents under `docs/superpowers/`; they remain recoverable from Git history. Experiment evidence records are retained separately.
- Removed three SHA-256-identical `root_*` metadata copies from `artifacts/2026-10-23/684869_wide20_recovery_metadata/`; the unprefixed originals remain.

## Verified redundant backup retained because deletion was blocked

`data/HipMRI_Study_open.local-backup-20261021/`: all 422 files matched the corresponding files beneath `data/HipMRI_Study_open/` by SHA-256 (1,402,579,880 bytes). Automatic approval blocked the shell deletion request. This backup remains on disk. No alternative deletion mechanism was used for this directory.

Historical experiment archives and source snapshots have not been classified as disposable merely because their filenames overlap. Unique checkpoints, predictions and final scheduler records remain retained.

## Local course export

Run `python scripts/build_course_submission.py --output local/course-submission-20261023` after staging the intended source files. The exporter uses the Git index file list and current working contents. It creates `recognition/HipMRI_3D_UNet_s4910970/` with the required Python modules and README at that folder's root, rewrites internal source paths, and excludes datasets, model files, archives, PDFs, internal plans and caches.

The export is a preview and contains no independent development history. Use the existing progressive history when transferring the contribution to the course fork. No PR, remote write or submission is performed by the exporter.

The development tree and flattened course export each passed all 63 tests. Four development Slurm scripts passed Bash syntax checks. A current nine-page A4 landscape README review PDF is retained at `local/COMP3710_README_review_20261023.pdf`; all nine rendered pages were inspected. It is a review export, not a Turnitin receipt.

## Requirements still needing human or external evidence

- Confirm the actual midpoint one-page review and tutor check-off. The README contains a retrospective summary, not a fabricated consultation record.
- Add anatomically verified prostate base/apex examples if required; current examples are deterministic high-error slices and are not described as confirmed anatomical levels.
- Review AI-assisted implementation and analysis personally before submission.
- Submit the reviewed contribution to upstream `topic-recognition`, export the final README to PDF for Turnitin, and respond to teaching-team review when released.

Exact remote dependency versions beyond retained environment records remain unavailable. No new GPU experiment was run for this repair.
