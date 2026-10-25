import hashlib
import json
from argparse import Namespace
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

import pytest

from modules import BaselineUNet3D, ImprovedUNet3D
from predict import build_test_metrics, load_checkpoint_model, mean_patient_prostate_dice, patient_prostate_dice, prostate_error_masks, prostate_error_slice_indices, render_case_views, run_prediction, sliding_window_logits


class ConstantLogitModel(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.ones_like(value), 2 * torch.ones_like(value)), dim=1)


def test_sliding_window_averages_overlap_and_covers_every_voxel():
    logits = sliding_window_logits(
        ConstantLogitModel(),
        torch.zeros(1, 7, 9, 10),
        patch_size=(4, 4, 4),
        stride=(3, 3, 3),
    )

    assert logits.shape == (2, 7, 9, 10)
    assert torch.allclose(logits[0], torch.ones(7, 9, 10))
    assert torch.allclose(logits[1], 2 * torch.ones(7, 9, 10))


def test_render_case_views_writes_multiplane_panel(tmp_path: Path):
    image = torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(8, 8, 8)
    label = torch.zeros(8, 8, 8, dtype=torch.long)
    prediction = label.clone()
    prediction[3:5, 3:5, 3:5] = 1
    output = tmp_path / "case.png"

    render_case_views(image, label, prediction, output, title="synthetic case", prostate_label=1)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_prostate_error_masks_include_cross_organ_misclassifications():
    """Catches the old background-only overlay that hid prostate-versus-organ errors."""

    label = torch.tensor([[[5, 2], [0, 5]]])
    prediction = torch.tensor([[[2, 5], [5, 5]]])

    false_positive, false_negative = prostate_error_masks(label, prediction, prostate_label=5)

    assert torch.equal(false_positive, torch.tensor([[[False, True], [True, False]]]))
    assert torch.equal(false_negative, torch.tensor([[[True, False], [False, False]]]))


def test_prostate_error_slice_indices_choose_the_highest_error_plane_not_volume_centre():
    """Catches panels that always inspected the central, potentially easy, slice."""

    label = torch.zeros(8, 8, 8, dtype=torch.long)
    prediction = label.clone()
    label[1, 2:6, 2:6] = 5
    prediction[1, 2:6, 2:6] = 2

    assert prostate_error_slice_indices(label, prediction, prostate_label=5) == (1, 2, 2)


def test_patient_prostate_dice_is_computed_per_patient_before_mean():
    patient_a_prediction = torch.tensor([[[1, 1], [0, 0]]])
    patient_a_target = torch.tensor([[[1, 0], [0, 0]]])
    patient_b_prediction = torch.tensor([[[1, 0], [0, 0]]])
    patient_b_target = torch.tensor([[[1, 1], [1, 1]]])

    scores = {
        "patient-a": patient_prostate_dice(patient_a_prediction, patient_a_target, prostate_label=1),
        "patient-b": patient_prostate_dice(patient_b_prediction, patient_b_target, prostate_label=1),
    }

    assert scores == {"patient-a": 2 / 3, "patient-b": 2 / 5}
    assert mean_patient_prostate_dice(scores) == pytest.approx((2 / 3 + 2 / 5) / 2)


def test_patient_prostate_dice_rejects_patient_without_reference_label():
    with pytest.raises(ValueError, match="no reference prostate voxels"):
        patient_prostate_dice(torch.zeros(2, 2, 2), torch.zeros(2, 2, 2), prostate_label=1)


def test_checkpoint_loader_restores_declared_baseline_model(tmp_path: Path):
    original = BaselineUNet3D(num_classes=2, base_channels=2)
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"model_state_dict": original.state_dict(), "config": {"model": "baseline", "num_classes": 2, "base_channels": 2}}, checkpoint_path)

    restored, config = load_checkpoint_model(checkpoint_path, torch.device("cpu"))

    assert isinstance(restored, BaselineUNet3D)
    assert config["model"] == "baseline"
    assert all(torch.equal(left, right) for left, right in zip(original.parameters(), restored.parameters(), strict=True))


def test_checkpoint_loader_rejects_context_weights_mislabeled_as_legacy(tmp_path: Path):
    """Missing version means legacy; Context weights must fail strict loading there."""
    original = ImprovedUNet3D(num_classes=2, base_channels=2)
    checkpoint_path = tmp_path / "legacy-improved.pt"
    torch.save({"model_state_dict": original.state_dict(), "config": {"model": "improved", "num_classes": 2, "base_channels": 2}}, checkpoint_path)

    with pytest.raises(RuntimeError):
        load_checkpoint_model(checkpoint_path, torch.device("cpu"))


def test_checkpoint_loader_restores_versioned_improved_checkpoint(tmp_path: Path):
    """A revised Improved checkpoint records and restores its exact architecture."""
    original = ImprovedUNet3D(num_classes=2, base_channels=2)
    checkpoint_path = tmp_path / "improved-v2.pt"
    torch.save({"model_state_dict": original.state_dict(), "config": {"model": "improved", "architecture_version": ImprovedUNet3D.architecture_version, "num_classes": 2, "base_channels": 2}}, checkpoint_path)

    restored, config = load_checkpoint_model(checkpoint_path, torch.device("cpu"))

    assert isinstance(restored, ImprovedUNet3D)
    assert config["architecture_version"] == ImprovedUNet3D.architecture_version
    assert all(torch.equal(left, right) for left, right in zip(original.parameters(), restored.parameters(), strict=True))


def test_test_metrics_report_patient_headline_and_nonbackground_classes():
    metrics = build_test_metrics(
        {"B006": 0.4, "C007": 0.8},
        {"0": {"dice": 1.0, "iou": 1.0}, "1": {"dice": 0.6, "iou": 0.5}, "2": {"dice": 0.7, "iou": 0.6}},
        prostate_label=1,
    )

    assert metrics["mean_patient_prostate_dice"] == pytest.approx(0.6)
    assert metrics["test_patient_count"] == 2
    assert set(metrics["per_class"]) == {"1", "2"}
    assert metrics["failure_ranking"] == [{"patient_id": "B006", "prostate_dice": 0.4}, {"patient_id": "C007", "prostate_dice": 0.8}]


def test_run_prediction_writes_test_only_metrics_and_ranked_failure_panel(tmp_path: Path):
    root = tmp_path / "data"
    test_image = root / "semantic_MRs" / "B003_Week0_LFOV.nii.gz"
    test_mask = root / "semantic_labels_only" / "B003_Week0_SEMANTIC.nii.gz"
    test_image.parent.mkdir(parents=True)
    test_mask.parent.mkdir(parents=True)
    image = np.arange(16**3, dtype=np.float32).reshape(16, 16, 16)
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[4:12, 4:12, 4:12] = 5
    nib.save(nib.Nifti1Image(image, np.eye(4)), test_image)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), test_mask)
    cases = [
        {"case_id": "B001_Week0", "patient_id": "B001", "image_path": "unused-image.nii.gz", "mask_path": "unused-mask.nii.gz", "shape": [16, 16, 16], "spacing": [1, 1, 1], "orientation": ["R", "A", "S"], "labels": [0, 5], "split": "train"},
        {"case_id": "B002_Week0", "patient_id": "B002", "image_path": "unused-image.nii.gz", "mask_path": "unused-mask.nii.gz", "shape": [16, 16, 16], "spacing": [1, 1, 1], "orientation": ["R", "A", "S"], "labels": [0, 5], "split": "validation"},
        {"case_id": "B003_Week0", "patient_id": "B003", "image_path": "semantic_MRs/B003_Week0_LFOV.nii.gz", "mask_path": "semantic_labels_only/B003_Week0_SEMANTIC.nii.gz", "shape": [16, 16, 16], "spacing": [1, 1, 1], "orientation": ["R", "A", "S"], "labels": [0, 5], "split": "test"},
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"summary": {"labels": [0, 1, 2, 3, 4, 5]}, "cases": cases}), encoding="utf-8")
    checkpoint = tmp_path / "best.pt"
    model = BaselineUNet3D(num_classes=6, base_channels=2)
    torch.save({"model_state_dict": model.state_dict(), "config": {"model": "baseline", "num_classes": 6, "base_channels": 2, "target_shape": [16, 16, 16], "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "prostate_label": 5, "prostate_label_source": "CSIRO, DOI: 10.25919/45T8-P065, Labels block: 5=Prostate."}}, checkpoint)

    output = tmp_path / "prediction"
    metrics = run_prediction(Namespace(checkpoint=checkpoint, manifest=manifest, data_root=root, output_dir=output, patch_size=(16, 16, 16), stride=(16, 16, 16), device="cpu"))

    saved = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
    assert saved["mean_patient_prostate_dice"] == metrics["mean_patient_prostate_dice"]
    assert saved["test_patient_count"] == 1
    assert set(saved["per_patient_prostate_dice"]) == {"B003"}
    assert saved["failure_ranking"][0]["patient_id"] == "B003"
    assert saved["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert saved["prostate_label"] == 5
    assert saved["peak_inference_vram_mb"] == 0.0
    assert saved["total_inference_seconds"] >= 0.0
    assert saved["mean_patient_inference_seconds"] >= 0.0
    assert saved["overlap"] == [0.0, 0.0, 0.0]
    assert saved["logit_merge_method"] == "overlap_averaged_logits"
    assert saved["failure_panel_selection"]["B003"]["case_id"] == "B003_Week0"
    assert saved["failure_panel_selection"]["B003"]["rule"] == "max_prostate_fp_plus_fn_per_plane; prostate_extent_centre_fallback"
    assert len(saved["failure_panel_selection"]["B003"]["slice_indices_dhw"]) == 3
    saved_prediction = torch.load(output / "predictions" / "B003_Week0.pt", map_location="cpu", weights_only=False)
    assert saved_prediction["case_id"] == "B003_Week0"
    assert saved_prediction["patient_id"] == "B003"
    assert saved_prediction["prediction"].dtype == torch.long
    assert saved_prediction["target"].dtype == torch.long
    assert saved_prediction["image"].shape == (1, 16, 16, 16)
    assert (output / "failure_panels" / "B003.png").is_file()
