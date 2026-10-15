import json
from pathlib import Path

import torch

from train import composite_3d_loss, evaluate_3d_batches, load_audited_pairs, select_validation_checkpoint


def test_composite_3d_loss_accepts_final_and_auxiliary_logits():
    target = torch.randint(0, 2, (1, 8, 8, 8))
    final = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    auxiliary = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
    loss = composite_3d_loss((final, (auxiliary,)), target, num_classes=2)
    loss.backward()
    assert torch.isfinite(loss)
    assert final.grad is not None and auxiliary.grad is not None


def test_evaluator_returns_per_class_metrics_and_resource_fields():
    model = torch.nn.Conv3d(1, 2, 1)
    batches = [(torch.randn(1, 1, 8, 8, 8), torch.randint(0, 2, (1, 8, 8, 8)))]
    result = evaluate_3d_batches(model, batches, torch.device("cpu"), num_classes=2)
    assert {"mean_dice", "mean_iou", "per_class", "inference_seconds", "peak_vram_mb"} <= set(result)


def test_checkpoint_selection_uses_validation_metric_only():
    history = [{"epoch": 1, "validation_mean_dice": 0.5, "test_mean_dice": 0.9}, {"epoch": 2, "validation_mean_dice": 0.7, "test_mean_dice": 0.1}]
    assert select_validation_checkpoint(history)["epoch"] == 2


def test_load_audited_pairs_restores_relative_case_metadata(tmp_path: Path):
    manifest = {
        "summary": {"labels": [0, 1]},
        "cases": [{"case_id": "B006_Week0", "patient_id": "B006", "image_path": "semantic_MRs/B006_Week0_LFOV.nii.gz", "mask_path": "semantic_labels_only/B006_Week0_SEMANTIC.nii.gz", "shape": [8, 8, 8], "spacing": [1.0, 1.0, 1.0], "orientation": ["L", "P", "S"], "labels": [0, 1], "split": "train"}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    pairs, split_ids, num_classes = load_audited_pairs(path)

    assert pairs[0].case_id == "B006_Week0"
    assert pairs[0].image_path == Path("semantic_MRs/B006_Week0_LFOV.nii.gz")
    assert split_ids == {"train": {"B006"}, "validation": set(), "test": set()}
    assert num_classes == 2
