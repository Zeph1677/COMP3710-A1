import json
from argparse import Namespace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from train import build_experiment_config, composite_3d_loss, create_argument_parser, evaluate_3d_batches, load_audited_pairs, run_overfit_preflight, run_training, select_validation_checkpoint


def test_composite_3d_loss_accepts_final_and_auxiliary_logits():
    target = torch.randint(0, 2, (1, 8, 8, 8))
    final = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    auxiliary = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
    loss = composite_3d_loss((final, (auxiliary,)), target, num_classes=2)
    loss.backward()
    assert torch.isfinite(loss)
    assert final.grad is not None and auxiliary.grad is not None


def test_composite_loss_penalizes_missed_prostate_more_when_weighted():
    target = torch.tensor([[[[1]]]])
    logits = torch.tensor([[[[[4.0]]], [[[0.0]]]]], requires_grad=True)

    unweighted = composite_3d_loss(logits, target, num_classes=2, prostate_label=1, prostate_ce_weight=1.0, prostate_dice_weight=1.0)
    weighted = composite_3d_loss(logits, target, num_classes=2, prostate_label=1, prostate_ce_weight=8.0, prostate_dice_weight=8.0)

    assert weighted > unweighted


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


def test_experiment_config_requires_cited_prostate_label_and_records_manifest_hash(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"summary": {"labels": [0, 1, 2]}}', encoding="utf-8")

    with pytest.raises(ValueError, match="prostate label source"):
        build_experiment_config(
            manifest_path=manifest,
            model_name="baseline",
            target_shape=(8, 8, 8),
            seed=42,
            prostate_label=1,
            prostate_label_source="",
        )

    config = build_experiment_config(
        manifest_path=manifest,
        model_name="baseline",
        target_shape=(8, 8, 8),
        seed=42,
        prostate_label=1,
        prostate_label_source="HipMRI documentation, Table 2",
    )

    assert config["prostate_label"] == 1
    assert config["prostate_label_source"] == "HipMRI documentation, Table 2"
    assert len(config["manifest_sha256"]) == 64


def test_experiment_config_rejects_out_of_range_prostate_label(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"summary": {"labels": [0, 1, 2]}}', encoding="utf-8")

    with pytest.raises(ValueError, match="outside manifest labels"):
        build_experiment_config(
            manifest_path=manifest,
            model_name="baseline",
            target_shape=(8, 8, 8),
            seed=42,
            prostate_label=3,
            prostate_label_source="HipMRI documentation, Table 2",
        )


def test_training_cli_requires_cited_mapping_and_accepts_explicit_geometry():
    parser = create_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--manifest", "manifest.json", "--data-root", "data"])

    args = parser.parse_args([
        "--manifest", "manifest.json", "--data-root", "data", "--prostate-label", "2",
        "--prostate-label-source", "HipMRI documentation, Table 2", "--target-shape", "32", "32", "32",
    ])

    assert args.model == "baseline"
    assert args.target_shape == [32, 32, 32]
    assert args.prostate_label == 2


def test_run_training_writes_reloadable_checkpoint_and_history(tmp_path: Path):
    root = tmp_path / "data"
    cases = []
    for split, patient in (("train", "B006"), ("validation", "C007"), ("test", "D008")):
        image_path = root / "semantic_MRs" / f"{patient}_Week0_LFOV.nii.gz"
        mask_path = root / "semantic_labels_only" / f"{patient}_Week0_SEMANTIC.nii.gz"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((16, 16, 16), dtype=np.float32)
        mask = np.zeros((16, 16, 16), dtype=np.int16)
        mask[4:12, 4:12, 4:12] = 1
        nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
        cases.append({"case_id": f"{patient}_Week0", "patient_id": patient, "image_path": str(image_path.relative_to(root)).replace("\\", "/"), "mask_path": str(mask_path.relative_to(root)).replace("\\", "/"), "shape": [16, 16, 16], "spacing": [1, 1, 1], "orientation": ["R", "A", "S"], "labels": [0, 1], "split": split})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"summary": {"labels": [0, 1]}, "cases": cases}), encoding="utf-8")
    output = tmp_path / "run"

    summary = run_training(Namespace(manifest=manifest, data_root=root, output_dir=output, model="baseline", epochs=1, batch_size=1, learning_rate=1e-3, base_channels=2, target_shape=(16, 16, 16), seed=42, device="cpu", prostate_label=1, prostate_label_source="Dataset documentation, Table 2"))

    assert summary["selected_epoch"] == 1.0
    assert (output / "experiment_config.json").is_file()
    assert (output / "history.json").is_file()
    assert (output / "training_summary.json").is_file()
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    saved_summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
    assert {"train_seconds", "validation_seconds", "train_peak_vram_mb", "validation_peak_vram_mb"} <= set(history[0])
    assert history[0]["train_peak_vram_mb"] == 0.0
    assert history[0]["validation_peak_vram_mb"] == 0.0
    assert {"peak_train_vram_mb", "peak_validation_vram_mb"} <= set(saved_summary)
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert checkpoint["config"]["manifest_sha256"]


def test_run_training_rejects_nonempty_output_directory(tmp_path: Path):
    output = tmp_path / "existing-run"
    output.mkdir()
    (output / "experiment_config.json").write_text("old evidence", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        run_training(Namespace(manifest=tmp_path / "missing.json", data_root=tmp_path, output_dir=output, model="baseline", epochs=1, batch_size=1, learning_rate=1e-3, base_channels=2, target_shape=(16, 16, 16), seed=42, device="cpu", prostate_label=1, prostate_label_source="Dataset documentation, Table 2"))


def test_overfit_preflight_rejects_non_train_patient_before_reading_data(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"summary": {"labels": [0, 1]}, "cases": [{"case_id": "C007_Week0", "patient_id": "C007", "image_path": "missing-image.nii.gz", "mask_path": "missing-mask.nii.gz", "shape": [16, 16, 16], "spacing": [1, 1, 1], "orientation": ["R", "A", "S"], "labels": [0, 1], "split": "validation"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="train split"):
        run_overfit_preflight(Namespace(manifest=manifest, data_root=tmp_path, output_dir=tmp_path / "preflight", model="baseline", epochs=1, batch_size=1, learning_rate=1e-3, base_channels=2, target_shape=(16, 16, 16), seed=42, device="cpu", prostate_label=1, prostate_label_source="Dataset documentation", preflight_patient_id="C007"))
