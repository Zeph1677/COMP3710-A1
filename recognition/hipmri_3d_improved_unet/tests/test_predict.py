from pathlib import Path

import torch

from predict import render_case_views, sliding_window_logits


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
