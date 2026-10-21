"""Read-only audit of foreground-patch exposure for one declared HipMRI case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import torch
from torch.nn import functional as F

# Retained as an injectable sentinel in the regression test: this audit must
# locate only the requested case and must never invoke full-dataset discovery.
from dataset import discover_volume_pairs

def _resized_mask(data_root: Path, case_id: str, target_shape: tuple[int, int, int]) -> torch.Tensor:
    mask_root = data_root / "semantic_labels_only"
    if not mask_root.is_dir():
        raise FileNotFoundError(f"required mask directory does not exist: {mask_root}")
    candidates = sorted(path for path in mask_root.rglob(f"{case_id}_SEMANTIC.nii*") if path.name in {f"{case_id}_SEMANTIC.nii", f"{case_id}_SEMANTIC.nii.gz"})
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one mask for {case_id}, found {len(candidates)}")
    raw = nib.load(candidates[0]).get_fdata(caching="unchanged")
    if raw.ndim != 3:
        raise ValueError(f"{case_id} mask must be three-dimensional")
    mask = torch.from_numpy(raw.astype("int64").transpose(2, 1, 0))
    return F.interpolate(mask[None, None].float(), size=target_shape, mode="nearest")[0, 0].long()


def _starts_for_anchor(point: torch.Tensor, shape: tuple[int, int, int], patch_shape: tuple[int, int, int], generator: torch.Generator) -> tuple[int, int, int]:
    starts = []
    for coordinate, size, patch in zip(point.tolist(), shape, patch_shape):
        low, high = max(0, coordinate - patch + 1), min(coordinate, size - patch)
        starts.append(int(torch.randint(low, high + 1, (), generator=generator).item()))
    return tuple(starts)


def _sample_report(mask: torch.Tensor, patch_shape: tuple[int, int, int], prostate_label: int, probability: float, samples: int, seed: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    coordinates = torch.nonzero(mask == prostate_label, as_tuple=False)
    rows: list[dict[str, object]] = []
    for _ in range(samples):
        anchored = bool(len(coordinates) and torch.rand((), generator=generator).item() < probability)
        if anchored:
            point = coordinates[torch.randint(len(coordinates), (), generator=generator)]
            starts = _starts_for_anchor(point, tuple(mask.shape), patch_shape, generator)
        else:
            starts = tuple(int(torch.randint(0, size - patch + 1, (), generator=generator).item()) for size, patch in zip(mask.shape, patch_shape))
        slices = tuple(slice(start, start + size) for start, size in zip(starts, patch_shape))
        voxels = int((mask[slices] == prostate_label).sum())
        rows.append({"foreground_anchor_attempted": anchored, "contains_prostate": bool(voxels), "prostate_voxels": voxels, "prostate_fraction": voxels / float(torch.tensor(patch_shape).prod()), "starts": list(starts)})

    def aggregate(group: list[dict[str, object]]) -> dict[str, object]:
        count = len(group)
        return {"sample_count": count, "contains_prostate_rate": sum(bool(row["contains_prostate"]) for row in group) / count if count else None, "mean_prostate_voxels": sum(int(row["prostate_voxels"]) for row in group) / count if count else None, "mean_prostate_fraction": sum(float(row["prostate_fraction"]) for row in group) / count if count else None}

    anchors = [row for row in rows if row["foreground_anchor_attempted"]]
    random_rows = [row for row in rows if not row["foreground_anchor_attempted"]]
    return {"samples": rows, "summary": aggregate(rows), "foreground_anchor": aggregate(anchors), "random": aggregate(random_rows)}


def audit_patch_coverage(data_root: Path | str, case_id: str, target_shape: tuple[int, int, int], patch_shape: tuple[int, int, int], prostate_label: int, foreground_probabilities: tuple[float, ...], samples: int, seed: int, output_path: Path | str) -> dict[str, object]:
    """Audit declared sampling probabilities without running training or inference."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if len(target_shape) != 3 or len(patch_shape) != 3 or any(patch <= 0 or patch > target for patch, target in zip(patch_shape, target_shape)):
        raise ValueError("patch shape must be three positive dimensions no larger than target shape")
    if any(not 0.0 <= probability <= 1.0 for probability in foreground_probabilities):
        raise ValueError("foreground probabilities must be in [0, 1]")
    root = Path(data_root).resolve()
    mask = _resized_mask(root, case_id, target_shape)
    report = {"case_id": case_id, "target_shape": list(target_shape), "patch_shape": list(patch_shape), "prostate_label": prostate_label, "samples_per_probability": samples, "seed": seed, "probabilities": {str(probability): _sample_report(mask, patch_shape, prostate_label, probability, samples, seed) for probability in foreground_probabilities}}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--case-id", default="B006_Week0")
    parser.add_argument("--target-shape", nargs=3, type=int, default=(128, 128, 128))
    parser.add_argument("--patch-shape", nargs=3, type=int, default=(96, 96, 96))
    parser.add_argument("--prostate-label", type=int, default=5)
    parser.add_argument("--probabilities", nargs="+", type=float, default=(0.5, 0.75))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_patch_coverage(args.data_root, args.case_id, tuple(args.target_shape), tuple(args.patch_shape), args.prostate_label, tuple(args.probabilities), args.samples, args.seed, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
