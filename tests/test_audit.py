import json
from pathlib import Path

import nibabel as nib
import numpy as np

from audit import write_dataset_manifest
from dataset import discover_volume_pairs


def _write_volume(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)


def _make_pair(root: Path, case_id: str) -> None:
    image = np.ones((6, 8, 10), dtype=np.float32)
    labels = np.zeros((6, 8, 10), dtype=np.uint8)
    labels[1:3, 2:5, 3:7] = 1
    _write_volume(root / "semantic_MRs" / f"{case_id}_LFOV.nii.gz", image)
    _write_volume(root / "semantic_labels_only" / f"{case_id}_SEMANTIC.nii.gz", labels)


def test_manifest_records_cases_relative_paths_labels_and_patient_disjoint_split(tmp_path: Path):
    for case_id in ("B006_Week0", "B006_Week1", "B037_Week0", "B038_Week0"):
        _make_pair(tmp_path, case_id)
    output = tmp_path / "dataset_manifest.json"

    manifest = write_dataset_manifest(discover_volume_pairs(tmp_path), output, seed=42)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == manifest
    assert manifest["summary"] == {
        "case_count": 4,
        "patient_count": 3,
        "labels": [0, 1],
        "split_case_counts": {"train": 1, "validation": 2, "test": 1},
        "split_patient_counts": {"train": 1, "validation": 1, "test": 1},
    }
    assert {record["case_id"] for record in manifest["cases"]} == {"B006_Week0", "B006_Week1", "B037_Week0", "B038_Week0"}
    assert all(not Path(record["image_path"]).is_absolute() for record in manifest["cases"])
    b006_splits = {record["split"] for record in manifest["cases"] if record["patient_id"] == "B006"}
    assert len(b006_splits) == 1
