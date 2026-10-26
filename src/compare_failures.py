"""Create deterministic, prostate-specific comparisons for matched formal runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import torch

from failure_selection import select_unique_patient_cases


_COMPATIBILITY_FIELDS = (
    "manifest_sha256",
    "prostate_label",
    "prostate_label_source",
    "target_shape",
    "patch_size",
    "stride",
)


def validate_comparison_inputs(
    baseline: dict[str, object], improved: dict[str, object]
) -> None:
    """Reject experiment artifacts that cannot support a fair comparison."""

    for field in _COMPATIBILITY_FIELDS:
        if baseline.get(field) != improved.get(field):
            raise ValueError(f"incompatible {field}")
    baseline_patients = set(baseline.get("per_patient_prostate_dice", {}))
    improved_patients = set(improved.get("per_patient_prostate_dice", {}))
    if baseline_patients != improved_patients:
        raise ValueError("incompatible test patient sets")


def common_error_score(
    baseline_counts: dict[str, int], improved_counts: dict[str, int]
) -> float:
    """Return mean whole-volume FP+FN over both model predictions."""

    return (
        baseline_counts["false_positive_voxels"]
        + baseline_counts["false_negative_voxels"]
        + improved_counts["false_positive_voxels"]
        + improved_counts["false_negative_voxels"]
    ) / 2.0


def error_relevant_prostate_slice(
    target: torch.Tensor,
    baseline: torch.Tensor,
    improved: torch.Tensor,
    prostate_label: int,
) -> int:
    """Choose the highest-error depth slice inside the reference extent."""

    extent = torch.nonzero(
        (target == prostate_label).any(dim=(1, 2)), as_tuple=False
    ).flatten()
    if not len(extent):
        raise ValueError("reference mask contains no prostate voxels")
    scores = []
    for index in extent.tolist():
        reference = target[index] == prostate_label
        errors = ((baseline[index] == prostate_label) ^ reference).sum()
        errors += ((improved[index] == prostate_label) ^ reference).sum()
        scores.append((int(errors), int(index)))
    return max(scores)[1]


def render_comparative_prostate_panel(
    image: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
    improved: torch.Tensor,
    prostate_label: int,
    output_path: Path | str,
) -> int:
    """Write one aligned six-panel image at the selected slice."""

    index = error_relevant_prostate_slice(
        target, baseline, improved, prostate_label
    )
    source = image[0] if image.ndim == 4 else image
    if source.shape != target.shape:
        raise ValueError("image and segmentation shapes must match")
    reference = target[index] == prostate_label
    baseline_errors = torch.zeros_like(target[index], dtype=torch.uint8)
    improved_errors = torch.zeros_like(target[index], dtype=torch.uint8)
    baseline_errors[(baseline[index] == prostate_label) & ~reference] = 1
    baseline_errors[(baseline[index] != prostate_label) & reference] = 2
    improved_errors[(improved[index] == prostate_label) & ~reference] = 1
    improved_errors[(improved[index] != prostate_label) & reference] = 2

    figure, axes = plt.subplots(1, 6, figsize=(18, 3))
    label_norm = BoundaryNorm([value - 0.5 for value in range(7)], 20)
    error_cmap = ListedColormap(["#252525", "#ffffff", "#e31a1c"])
    error_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    panels = (
        (source[index], "MRI", "gray"),
        (target[index], "Ground truth", "tab20"),
        (baseline[index], "Baseline", "tab20"),
        (improved[index], "Improved", "tab20"),
        (baseline_errors, "Baseline FP/FN", error_cmap),
        (improved_errors, "Improved FP/FN", error_cmap),
    )
    for number, (axis, (data, title, colour_map)) in enumerate(zip(axes, panels)):
        options = {} if number == 0 else {"norm": label_norm}
        if number >= 4:
            options = {"norm": error_norm}
        axis.imshow(data.numpy(), cmap=colour_map, interpolation="nearest", **options)
        axis.set_title(title)
        axis.axis("off")
    figure.legend(
        handles=[
            Patch(facecolor="#252525", label="No prostate error"),
            Patch(facecolor="#ffffff", edgecolor="black", label="FP"),
            Patch(facecolor="#e31a1c", label="FN"),
        ],
        loc="lower center",
        ncol=3,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return index


def _counts(
    prediction: torch.Tensor, target: torch.Tensor, label: int
) -> dict[str, int]:
    predicted = prediction == label
    reference = target == label
    true_positive = int((predicted & reference).sum())
    return {
        "true_positive_voxels": true_positive,
        "false_positive_voxels": int(predicted.sum()) - true_positive,
        "false_negative_voxels": int(reference.sum()) - true_positive,
    }


def rank_failure_cases(
    baseline_predictions: dict[str, dict[str, object]],
    improved_predictions: dict[str, dict[str, object]],
    prostate_label: int,
) -> list[dict[str, object]]:
    """Rank common scans by shared voxel error with deterministic ties."""

    if set(baseline_predictions) != set(improved_predictions):
        raise ValueError("incompatible prediction case sets")
    ranking = []
    for case_id in sorted(baseline_predictions):
        baseline = baseline_predictions[case_id]
        improved = improved_predictions[case_id]
        for field in (
            "case_id",
            "patient_id",
            "manifest_sha256",
            "prostate_label",
            "target_shape",
        ):
            if baseline.get(field) != improved.get(field):
                raise ValueError(f"incompatible prediction {field} for {case_id}")
        if baseline.get("case_id") != case_id:
            raise ValueError(f"prediction case ID does not match filename for {case_id}")
        if baseline.get("prostate_label") != prostate_label:
            raise ValueError(
                f"prediction prostate label does not match comparison label for {case_id}"
            )
        if not torch.equal(baseline["target"], improved["target"]):
            raise ValueError(f"incompatible case payload for {case_id}")
        baseline_counts = _counts(
            baseline["prediction"], baseline["target"], prostate_label
        )
        improved_counts = _counts(
            improved["prediction"], improved["target"], prostate_label
        )
        ranking.append(
            {
                "case_id": case_id,
                "patient_id": baseline["patient_id"],
                "error_score": common_error_score(
                    baseline_counts, improved_counts
                ),
                "slice_index": error_relevant_prostate_slice(
                    baseline["target"],
                    baseline["prediction"],
                    improved["prediction"],
                    prostate_label,
                ),
                "baseline": baseline_counts,
                "improved": improved_counts,
            }
        )
    return sorted(
        ranking,
        key=lambda item: (
            -float(item["error_score"]),
            str(item["patient_id"]),
            str(item["case_id"]),
        ),
    )


def build_selection_manifest(
    selected_cases: list[dict[str, object]], git_sha: str
) -> dict[str, object]:
    """Preserve the selection rule together with the code identity."""

    if not git_sha.strip():
        raise ValueError("selection manifest requires a Git SHA")
    return {
        "rule_version": 2,
        "git_sha": git_sha,
        "rule": (
            "rank scans by mean whole-volume prostate FP+FN across baseline and "
            "improved; retain the highest-ranked scan per patient; patient_id then "
            "case_id tie break; highest-error slice inside GT prostate extent"
        ),
        "selected_cases": selected_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--improved-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-patients", type=int, default=5)
    args = parser.parse_args()

    baseline_metrics = json.loads(
        (args.baseline_dir / "test_metrics.json").read_text(encoding="utf-8")
    )
    improved_metrics = json.loads(
        (args.improved_dir / "test_metrics.json").read_text(encoding="utf-8")
    )
    validate_comparison_inputs(baseline_metrics, improved_metrics)

    def load(directory: Path) -> dict[str, dict[str, object]]:
        return {
            path.stem: torch.load(path, map_location="cpu", weights_only=False)
            for path in (directory / "predictions").glob("*.pt")
        }

    baseline_predictions = load(args.baseline_dir)
    improved_predictions = load(args.improved_dir)
    ranking = rank_failure_cases(
        baseline_predictions,
        improved_predictions,
        int(baseline_metrics["prostate_label"]),
    )
    selected = select_unique_patient_cases(ranking, args.max_patients)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for entry in selected:
        case_id = str(entry["case_id"])
        payload = baseline_predictions[case_id]
        panel = Path("comparison_panels") / f"{entry['patient_id']}__{case_id}.png"
        entry["panel"] = panel.as_posix()
        entry["slice_index"] = render_comparative_prostate_panel(
            payload["image"],
            payload["target"],
            payload["prediction"],
            improved_predictions[case_id]["prediction"],
            int(baseline_metrics["prostate_label"]),
            args.output_dir / panel,
        )
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = build_selection_manifest(selected, git_sha)
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
