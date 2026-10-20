from pathlib import Path

import torch

import pytest

from predict import mean_patient_prostate_dice, patient_prostate_dice, render_case_views, sliding_window_logits


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
