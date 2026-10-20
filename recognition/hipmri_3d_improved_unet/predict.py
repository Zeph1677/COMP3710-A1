"""Patch aggregation and multi-plane case figures for held-out HipMRI evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from modules import BaselineUNet3D, ImprovedUNet3D


def patient_prostate_dice(prediction: torch.Tensor, target: torch.Tensor, prostate_label: int) -> float:
    """Compute Dice for one reconstructed patient volume, never pooled across patients."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    predicted = prediction == prostate_label
    actual = target == prostate_label
    actual_count = int(actual.sum())
    if not actual_count:
        raise ValueError("patient has no reference prostate voxels")
    denominator = int(predicted.sum()) + actual_count
    return 2 * int((predicted & actual).sum()) / denominator if denominator else 0.0


def mean_patient_prostate_dice(per_patient_scores: dict[str, float]) -> float:
    """Return the arithmetic mean of independently reconstructed patient Dice values."""

    if not per_patient_scores:
        raise ValueError("at least one patient prostate Dice is required")
    return sum(per_patient_scores.values()) / len(per_patient_scores)


def load_checkpoint_model(checkpoint_path: Path | str, device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    """Restore exactly the architecture declared in a validation-selected checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing experiment config")
    model_name, num_classes, base_channels = config.get("model"), config.get("num_classes"), config.get("base_channels")
    if model_name not in {"baseline", "improved"} or not isinstance(num_classes, int) or not isinstance(base_channels, int):
        raise ValueError("checkpoint config does not declare a supported model")
    model_type = BaselineUNet3D if model_name == "baseline" else ImprovedUNet3D
    model = model_type(num_classes=num_classes, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def _window_starts(length: int, patch: int, stride: int) -> list[int]:
    if patch <= 0 or stride <= 0:
        raise ValueError("patch dimensions and strides must be positive")
    if patch > length:
        raise ValueError(f"patch dimension {patch} exceeds volume dimension {length}")
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def sliding_window_logits(
    model: torch.nn.Module,
    volume: torch.Tensor,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> torch.Tensor:
    """Average overlapping patch logits and assert every output voxel was predicted."""

    if volume.ndim != 4:
        raise ValueError(f"volume must have shape (C, D, H, W), found {tuple(volume.shape)}")
    starts = [_window_starts(length, patch, step) for length, patch, step in zip(volume.shape[-3:], patch_size, stride, strict=True)]
    previous_training = model.training
    model.eval()
    logits_sum: torch.Tensor | None = None
    prediction_count = torch.zeros(volume.shape[-3:], dtype=torch.int32, device=volume.device)
    with torch.no_grad():
        for depth in starts[0]:
            for height in starts[1]:
                for width in starts[2]:
                    patch = volume[:, depth:depth + patch_size[0], height:height + patch_size[1], width:width + patch_size[2]]
                    output = model(patch[None])
                    logits = output[0] if isinstance(output, tuple) else output
                    if logits_sum is None:
                        logits_sum = torch.zeros((logits.shape[1], *volume.shape[-3:]), dtype=logits.dtype, device=logits.device)
                    logits_sum[:, depth:depth + patch_size[0], height:height + patch_size[1], width:width + patch_size[2]] += logits[0]
                    prediction_count[depth:depth + patch_size[0], height:height + patch_size[1], width:width + patch_size[2]] += 1
    model.train(previous_training)
    if logits_sum is None or torch.any(prediction_count == 0):
        raise RuntimeError("sliding-window inference did not produce a prediction for every voxel")
    return logits_sum / prediction_count.to(dtype=logits_sum.dtype).unsqueeze(0)


def _as_volume(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"case views require a 3D array, found {array.shape}")
    return array


def render_case_views(
    image: torch.Tensor | np.ndarray,
    label: torch.Tensor | np.ndarray,
    prediction: torch.Tensor | np.ndarray,
    output_path: Path | str,
    title: str,
) -> None:
    """Render axial, coronal and sagittal MRI/label/prediction/error panels."""

    image_array, label_array, prediction_array = (_as_volume(value) for value in (image, label, prediction))
    if image_array.shape != label_array.shape or image_array.shape != prediction_array.shape:
        raise ValueError("image, label and prediction must have identical 3D shapes")
    centre = tuple(dimension // 2 for dimension in image_array.shape)
    planes = (
        ("axial", image_array[centre[0]], label_array[centre[0]], prediction_array[centre[0]]),
        ("coronal", image_array[:, centre[1], :], label_array[:, centre[1], :], prediction_array[:, centre[1], :]),
        ("sagittal", image_array[:, :, centre[2]], label_array[:, :, centre[2]], prediction_array[:, :, centre[2]]),
    )
    figure, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    for row, (plane_name, image_plane, label_plane, prediction_plane) in enumerate(planes):
        false_negative = (label_plane != 0) & (prediction_plane == 0)
        false_positive = (label_plane == 0) & (prediction_plane != 0)
        panels = (("MRI", image_plane, "gray"), ("ground truth", label_plane, "tab20"), ("prediction", prediction_plane, "tab20"), ("FP / FN", false_positive.astype(int) + 2 * false_negative.astype(int), "coolwarm"))
        for column, (name, panel, colour_map) in enumerate(panels):
            axes[row, column].imshow(panel.T, cmap=colour_map, origin="lower")
            axes[row, column].set_title(f"{plane_name}: {name}")
            axes[row, column].axis("off")
    figure.suptitle(title)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
