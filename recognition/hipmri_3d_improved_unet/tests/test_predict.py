from pathlib import Path

import torch

import pytest

from modules import BaselineUNet3D
from predict import build_test_metrics, load_checkpoint_model, mean_patient_prostate_dice, patient_prostate_dice, render_case_views, sliding_window_logits


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

    render_case_views(image, label, prediction, output, title="synthetic case")

    assert output.is_file()
    assert output.stat().st_size > 0


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
