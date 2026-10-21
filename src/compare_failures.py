"""Create deterministic, prostate-specific comparisons for matched formal runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import torch


_COMPATIBILITY_FIELDS = ("manifest_sha256", "prostate_label", "prostate_label_source", "target_shape", "patch_size", "stride")


def validate_comparison_inputs(baseline: dict[str, object], improved: dict[str, object]) -> None:
    """Reject experiment artifacts that cannot support a fair same-case comparison."""

    for field in _COMPATIBILITY_FIELDS:
        if baseline.get(field) != improved.get(field):
            raise ValueError(f"incompatible {field}")
    if set(baseline.get("per_patient_prostate_dice", {})) != set(improved.get("per_patient_prostate_dice", {})):
        raise ValueError("incompatible test patient sets")


def common_error_score(baseline_counts: dict[str, int], improved_counts: dict[str, int]) -> float:
    """Return the predeclared mean FP/FN count over the two model predictions."""

    return (baseline_counts["false_positive_voxels"] + baseline_counts["false_negative_voxels"] + improved_counts["false_positive_voxels"] + improved_counts["false_negative_voxels"]) / 2.0


def error_relevant_prostate_slice(target: torch.Tensor, baseline: torch.Tensor, improved: torch.Tensor, prostate_label: int) -> int:
    """Choose the highest-error depth slice within the ground-truth prostate extent."""

    extent = torch.nonzero((target == prostate_label).any(dim=(1, 2)), as_tuple=False).flatten()
    if not len(extent):
        raise ValueError("reference mask contains no prostate voxels")
    scores = []
    for index in extent.tolist():
        reference = target[index] == prostate_label
        errors = ((baseline[index] == prostate_label) ^ reference).sum() + ((improved[index] == prostate_label) ^ reference).sum()
        scores.append((int(errors), int(index)))
    return max(scores)[1]


def render_comparative_prostate_panel(image: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor, improved: torch.Tensor, prostate_label: int, output_path: Path | str) -> int:
    """Write one aligned six-panel image at the declared error-relevant slice."""

    index = error_relevant_prostate_slice(target, baseline, improved, prostate_label)
    source = image[0] if image.ndim == 4 else image
    if source.shape != target.shape:
        raise ValueError("image and segmentation shapes must match")
    reference = target[index] == prostate_label
    baseline_fp_fn = torch.zeros_like(target[index], dtype=torch.uint8)
    improved_fp_fn = torch.zeros_like(target[index], dtype=torch.uint8)
    baseline_fp_fn[(baseline[index] == prostate_label) & ~reference] = 1
    baseline_fp_fn[(baseline[index] != prostate_label) & reference] = 2
    improved_fp_fn[(improved[index] == prostate_label) & ~reference] = 1
    improved_fp_fn[(improved[index] != prostate_label) & reference] = 2
    figure, axes = plt.subplots(1, 6, figsize=(18, 3))
    panels = ((source[index], "MRI", "gray"), (target[index], "Ground truth", "tab20"), (baseline[index], "Baseline", "tab20"), (improved[index], "Improved", "tab20"), (baseline_fp_fn, "Baseline FP/FN", "coolwarm"), (improved_fp_fn, "Improved FP/FN", "coolwarm"))
    for axis, (data, title, cmap) in zip(axes, panels):
        axis.imshow(data.numpy(), cmap=cmap, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return index


def _counts(prediction: torch.Tensor, target: torch.Tensor, label: int) -> dict[str, int]:
    predicted, reference = prediction == label, target == label
    true_positive = int((predicted & reference).sum())
    return {"true_positive_voxels": true_positive, "false_positive_voxels": int(predicted.sum()) - true_positive, "false_negative_voxels": int(reference.sum()) - true_positive}


def rank_failure_cases(baseline_predictions: dict[str, dict[str, object]], improved_predictions: dict[str, dict[str, object]], prostate_label: int) -> list[dict[str, object]]:
    """Rank common cases by declared shared voxel error with patient-ID tie breaking."""

    if set(baseline_predictions) != set(improved_predictions):
        raise ValueError("incompatible prediction case sets")
    ranking = []
    for case_id in sorted(baseline_predictions):
        baseline, improved = baseline_predictions[case_id], improved_predictions[case_id]
        for field in ("case_id", "patient_id", "manifest_sha256", "prostate_label", "target_shape"):
            if baseline.get(field) != improved.get(field):
                raise ValueError(f"incompatible prediction {field} for {case_id}")
        if baseline.get("case_id") != case_id:
            raise ValueError(f"prediction case ID does not match filename for {case_id}")
        if baseline.get("prostate_label") != prostate_label:
            raise ValueError(f"prediction prostate label does not match comparison label for {case_id}")
        if not torch.equal(baseline["target"], improved["target"]):
            raise ValueError(f"incompatible case payload for {case_id}")
        baseline_counts = _counts(baseline["prediction"], baseline["target"], prostate_label)
        improved_counts = _counts(improved["prediction"], improved["target"], prostate_label)
        ranking.append({"case_id": case_id, "patient_id": baseline["patient_id"], "error_score": common_error_score(baseline_counts, improved_counts), "slice_index": error_relevant_prostate_slice(baseline["target"], baseline["prediction"], improved["prediction"], prostate_label), "baseline": baseline_counts, "improved": improved_counts})
    return sorted(ranking, key=lambda item: (-float(item["error_score"]), str(item["patient_id"]), str(item["case_id"])))


def build_selection_manifest(selected_cases: list[dict[str, object]], git_sha: str) -> dict[str, object]:
    """Preserve the deterministic selection rule together with the code identity."""

    if not git_sha.strip():
        raise ValueError("selection manifest requires a Git SHA")
    return {"rule_version": 1, "git_sha": git_sha, "rule": "mean FP plus FN across baseline and improved; patient_id then case_id tie break; highest-error slice inside GT prostate extent", "selected_cases": selected_cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--improved-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=5)
    args = parser.parse_args()
    baseline_metrics = json.loads((args.baseline_dir / "test_metrics.json").read_text(encoding="utf-8"))
    improved_metrics = json.loads((args.improved_dir / "test_metrics.json").read_text(encoding="utf-8"))
    validate_comparison_inputs(baseline_metrics, improved_metrics)
    load = lambda directory: {path.stem: torch.load(path, map_location="cpu", weights_only=False) for path in (directory / "predictions").glob("*.pt")}
    baseline_predictions, improved_predictions = load(args.baseline_dir), load(args.improved_dir)
    ranking = rank_failure_cases(baseline_predictions, improved_predictions, int(baseline_metrics["prostate_label"]))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected = ranking[:args.max_cases]
    for entry in selected:
        case_id = str(entry["case_id"])
        payload = baseline_predictions[case_id]
        entry["panel"] = str(Path("comparison_panels") / f"{entry['patient_id']}__{case_id}.png")
        entry["slice_index"] = render_comparative_prostate_panel(payload["image"], payload["target"], payload["prediction"], improved_predictions[case_id]["prediction"], int(baseline_metrics["prostate_label"]), args.output_dir / entry["panel"])
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    manifest = build_selection_manifest(selected, git_sha)
    (args.output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
