import json
from pathlib import Path

import pytest

from plot_training import load_history, render_training_curves, selected_epoch


def test_load_history_accepts_training_summary_object(tmp_path: Path):
    history_path = tmp_path / "training_summary.json"
    history_path.write_text(json.dumps({"history": [{"epoch": 1, "train_loss": 1.2, "validation_mean_dice": 0.4}]}), encoding="utf-8")

    assert load_history(history_path) == [{"epoch": 1, "train_loss": 1.2, "validation_mean_dice": 0.4}]


def test_selected_epoch_uses_highest_saved_validation_mean_dice():
    history = [
        {"epoch": 1, "train_loss": 1.0, "validation_mean_dice": 0.70},
        {"epoch": 2, "train_loss": 0.8, "validation_mean_dice": 0.82},
        {"epoch": 3, "train_loss": 0.6, "validation_mean_dice": 0.81},
    ]

    assert selected_epoch(history, "mean_dice") == 2


def test_selected_epoch_rejects_missing_prostate_selection_metrics():
    history = [{"epoch": 1, "train_loss": 1.0, "validation_mean_dice": 0.70}]

    with pytest.raises(ValueError, match="prostate"):
        selected_epoch(history, "prostate_dice")


def test_render_training_curves_writes_nonempty_png(tmp_path: Path):
    histories = {
        "Baseline": [
            {"epoch": 1, "train_loss": 1.0, "validation_mean_dice": 0.60},
            {"epoch": 2, "train_loss": 0.7, "validation_mean_dice": 0.75},
        ],
        "Improved": [
            {"epoch": 1, "train_loss": 1.1, "validation_mean_dice": 0.62},
            {"epoch": 2, "train_loss": 0.8, "validation_mean_dice": 0.74},
        ],
    }
    output = tmp_path / "curves.png"

    render_training_curves(histories, output, selection_metric="mean_dice")

    assert output.is_file()
    assert output.stat().st_size > 0
