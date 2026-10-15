"""Training utilities for fair HipMRI 3D baseline-versus-improved experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import torch
from torch.nn import functional as F

from dataset import VolumePair


def load_audited_pairs(manifest_path: Path | str) -> tuple[list[VolumePair], dict[str, set[str]], int]:
    """Restore only audited relative metadata and the fixed patient partitions."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    labels = manifest.get("summary", {}).get("labels")
    cases = manifest.get("cases")
    if not isinstance(labels, list) or labels != list(range(len(labels))) or not cases:
        raise ValueError("manifest must contain contiguous labels and at least one audited case")
    pairs = [
        VolumePair(
            case_id=case["case_id"], patient_id=case["patient_id"], image_path=Path(case["image_path"]), mask_path=Path(case["mask_path"]),
            shape=tuple(case["shape"]), spacing=tuple(case["spacing"]), orientation=tuple(case["orientation"]), labels=tuple(case["labels"]),
        )
        for case in cases
    ]
    split_ids = {split: {case["patient_id"] for case in cases if case["split"] == split} for split in ("train", "validation", "test")}
    if set().union(*split_ids.values()) != {pair.patient_id for pair in pairs}:
        raise ValueError("manifest has unassigned patients")
    if split_ids["train"] & split_ids["validation"] or split_ids["train"] & split_ids["test"] or split_ids["validation"] & split_ids["test"]:
        raise ValueError("manifest patient partitions overlap")
    return pairs, split_ids, len(labels)


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target, num_classes).movedim(-1, 1).float()
    overlap = (probabilities * one_hot).sum(dim=(0, 2, 3, 4))
    total = probabilities.sum(dim=(0, 2, 3, 4)) + one_hot.sum(dim=(0, 2, 3, 4))
    return 1 - ((2 * overlap + 1e-6) / (total + 1e-6)).mean()


def composite_3d_loss(output: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]], target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Cross entropy plus soft Dice, with explicitly downsampled auxiliary targets."""

    final, auxiliaries = output if isinstance(output, tuple) else (output, ())
    loss = F.cross_entropy(final, target) + _soft_dice_loss(final, target, num_classes)
    for auxiliary in auxiliaries:
        scaled_target = F.interpolate(target[:, None].float(), size=auxiliary.shape[-3:], mode="nearest")[:, 0].long()
        loss = loss + 0.25 * (F.cross_entropy(auxiliary, scaled_target) + _soft_dice_loss(auxiliary, scaled_target, num_classes))
    return loss


def evaluate_3d_batches(model: torch.nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device, num_classes: int) -> dict[str, object]:
    """Compute hard held-out Dice/IoU and resource fields without model selection."""

    model.eval()
    prediction_parts, target_parts = [], []
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for images, masks in batches:
            output = model(images.to(device))
            logits = output[0] if isinstance(output, tuple) else output
            prediction_parts.append(logits.argmax(1).cpu()); target_parts.append(masks.cpu())
    if device.type == "cuda": torch.cuda.synchronize(device)
    prediction, target = torch.cat(prediction_parts), torch.cat(target_parts)
    per_class, dice_values, iou_values = {}, [], []
    for label in range(num_classes):
        predicted, actual = prediction == label, target == label
        intersection = int((predicted & actual).sum())
        union = int((predicted | actual).sum())
        denominator = int(predicted.sum() + actual.sum())
        dice = (2 * intersection / denominator) if denominator else None
        iou = (intersection / union) if union else None
        per_class[str(label)] = {"dice": dice, "iou": iou, "voxels": int(actual.sum())}
        if dice is not None: dice_values.append(dice)
        if iou is not None: iou_values.append(iou)
    return {"per_class": per_class, "mean_dice": sum(dice_values) / len(dice_values), "mean_iou": sum(iou_values) / len(iou_values), "inference_seconds": time.perf_counter() - started, "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0}


def select_validation_checkpoint(history: list[dict[str, float]]) -> dict[str, float]:
    """Choose solely by saved validation Dice; test values cannot influence selection."""

    if not history: raise ValueError("cannot select checkpoint from empty history")
    return max(history, key=lambda row: row["validation_mean_dice"])
