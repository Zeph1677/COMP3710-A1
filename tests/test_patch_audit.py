import json
from pathlib import Path

import nibabel as nib
import numpy as np

from patch_audit import audit_patch_coverage


def _write_pair(root: Path) -> None:
    image_path = root / "semantic_MRs" / "B006_Week0_LFOV.nii.gz"
    mask_path = root / "semantic_labels_only" / "B006_Week0_SEMANTIC.nii.gz"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    image = np.zeros((16, 16, 16), dtype=np.float32)
    mask = np.zeros((16, 16, 16), dtype=np.int16)
    mask[5:9, 5:9, 5:9] = 5
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)


def test_patch_audit_reads_only_the_declared_case_not_the_entire_dataset(tmp_path: Path, monkeypatch):
    _write_pair(tmp_path / "data")
    output = tmp_path / "audit.json"
    monkeypatch.setattr("patch_audit.discover_volume_pairs", lambda _: (_ for _ in ()).throw(AssertionError("full dataset discovery is not permitted")))

    audit_patch_coverage(
        data_root=tmp_path / "data",
        case_id="B006_Week0",
        target_shape=(16, 16, 16),
        patch_shape=(8, 8, 8),
        prostate_label=5,
        foreground_probabilities=(0.5, 0.75),
        samples=24,
        seed=42,
        output_path=output,
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["case_id"] == "B006_Week0"
    assert saved["target_shape"] == [16, 16, 16]
    assert saved["patch_shape"] == [8, 8, 8]
    assert set(saved["probabilities"]) == {"0.5", "0.75"}
    for report in saved["probabilities"].values():
        assert len(report["samples"]) == 24
        assert {"foreground_anchor_attempted", "contains_prostate", "prostate_voxels", "prostate_fraction"} <= set(report["samples"][0])
        assert report["foreground_anchor"]["contains_prostate_rate"] == 1.0
        assert report["summary"]["sample_count"] == 24
