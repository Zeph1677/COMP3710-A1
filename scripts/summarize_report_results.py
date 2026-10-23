"""Create a traceable, Git-safe summary from retained HipMRI result records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        raise ValueError(f"required source file is missing: {relative_path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected in: {relative_path}")
    return value


def _source_file(root: Path, relative_path: str) -> dict[str, str]:
    path = root / relative_path
    if not path.is_file():
        raise ValueError(f"required source file is missing: {relative_path}")
    return {"path": relative_path.replace("\\", "/"), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _require_equal(group_id: str, field: str, baseline: object, improved: object) -> object:
    if baseline != improved:
        raise ValueError(f"{group_id}: protocol mismatch for {field}: {baseline!r} != {improved!r}")
    return baseline


def _required(mapping: dict[str, Any], field: str, source_name: str) -> Any:
    value = mapping.get(field)
    if value is None:
        raise ValueError(f"{source_name} is missing required field: {field}")
    return value


def _load_arm(root: Path, paths: dict[str, str]) -> dict[str, Any]:
    config = _read_json(root, paths["config"])
    training_summary = _read_json(root, paths["training_summary"])
    metrics = _read_json(root, paths["metrics"])
    git_sha = (root / paths["git_sha"]).read_text(encoding="utf-8").strip()
    if not git_sha:
        raise ValueError(f"git SHA is empty: {paths['git_sha']}")

    patient_scores = _required(metrics, "per_patient_prostate_dice", paths["metrics"])
    if not isinstance(patient_scores, dict) or not patient_scores:
        raise ValueError(f"{paths['metrics']} has no per-patient prostate Dice mapping")
    patient_scores = {str(patient_id): float(score) for patient_id, score in patient_scores.items()}
    recomputed_mean = sum(patient_scores.values()) / len(patient_scores)
    reported_mean = float(_required(metrics, "mean_patient_prostate_dice", paths["metrics"]))
    if abs(recomputed_mean - reported_mean) > 1e-12:
        raise ValueError(f"{paths['metrics']}: reported patient mean differs from per-patient mean")

    missing_fields: list[str] = []
    resource_metrics: dict[str, float | int | None] = {}
    for field, source in (
        ("parameter_count", training_summary),
        ("peak_train_vram_mb", training_summary),
        ("peak_inference_vram_mb", metrics),
        ("mean_patient_inference_seconds", metrics),
    ):
        value = source.get(field)
        if value is None:
            missing_fields.append(field)
        resource_metrics[field] = value

    return {
        "model": _required(config, "model", paths["config"]),
        "git_sha": git_sha,
        "manifest_sha256": _required(metrics, "manifest_sha256", paths["metrics"]),
        "prostate_label": _required(metrics, "prostate_label", paths["metrics"]),
        "prostate_label_source": _required(metrics, "prostate_label_source", paths["metrics"]),
        "test_protocol": {
            "patch_size": _required(metrics, "patch_size", paths["metrics"]),
            "stride": _required(metrics, "stride", paths["metrics"]),
            "logit_merge_method": _required(metrics, "logit_merge_method", paths["metrics"]),
        },
        "selected_epoch": training_summary.get("selected_epoch"),
        "patient_scores": patient_scores,
        "mean_patient_prostate_dice": recomputed_mean,
        "resource_metrics": resource_metrics,
        "missing_fields": missing_fields,
        "source_files": {name: _source_file(root, path) for name, path in paths.items()},
    }


def _validate_group(group_id: str, baseline: dict[str, Any], improved: dict[str, Any]) -> None:
    for field in ("manifest_sha256", "prostate_label", "prostate_label_source", "test_protocol"):
        _require_equal(group_id, field, baseline[field], improved[field])
    if set(baseline["patient_scores"]) != set(improved["patient_scores"]):
        raise ValueError(f"{group_id}: protocol mismatch for patient_ids")


def build_report_summary(root: Path | str, groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize independently declared paired experiment groups without mixing them."""

    root = Path(root)
    summaries: list[dict[str, Any]] = []
    for group in groups:
        group_id = str(group["group_id"])
        baseline = _load_arm(root, group["baseline"])
        improved = _load_arm(root, group["improved"])
        _validate_group(group_id, baseline, improved)
        paired = [
            {
                "patient_id": patient_id,
                "baseline": baseline["patient_scores"][patient_id],
                "improved": improved["patient_scores"][patient_id],
                "improved_minus_baseline": improved["patient_scores"][patient_id] - baseline["patient_scores"][patient_id],
            }
            for patient_id in sorted(baseline["patient_scores"])
        ]
        summaries.append({
            "group_id": group_id,
            "selection_metric": group["selection_metric"],
            "manifest_sha256": baseline["manifest_sha256"],
            "job_ids": group["job_ids"] if "job_ids" in group else None,
            "prostate_label": baseline["prostate_label"],
            "prostate_label_source": baseline["prostate_label_source"],
            "test_protocol": baseline["test_protocol"],
            "arms": {"baseline": baseline, "improved": improved},
            "headline": {
                "baseline_mean_patient_prostate_dice": baseline["mean_patient_prostate_dice"],
                "improved_mean_patient_prostate_dice": improved["mean_patient_prostate_dice"],
                "improved_minus_baseline": improved["mean_patient_prostate_dice"] - baseline["mean_patient_prostate_dice"],
            },
            "paired_patient_dice": paired,
        })
    return {"schema_version": 1, "experiment_groups": summaries}


def ensure_output_is_not_source(root: Path | str, output: Path | str, groups: list[dict[str, Any]]) -> None:
    """Prevent an explicit output path from replacing retained evidence inputs."""

    root = Path(root).resolve()
    output_path = (root / output).resolve()
    source_paths = {
        (root / relative_path).resolve()
        for group in groups
        for arm_name in ("baseline", "improved")
        for relative_path in group[arm_name].values()
    }
    if output_path in source_paths:
        raise ValueError(f"refuses to overwrite declared source file: {output}")


def _default_groups() -> list[dict[str, Any]]:
    return [
        {
            "group_id": "main_matched_formal_4232e0d",
            "selection_metric": "validation_mean_dice",
            "job_ids": {"baseline": "683675", "improved": "683676"},
            "baseline": {
                "config": "artifacts/2026-10-22/683675_formal_baseline/experiment_config.json",
                "training_summary": "artifacts/2026-10-22/683675_formal_baseline/training_summary.json",
                "metrics": "artifacts/2026-10-22/683675_formal_baseline/test_metrics.json",
                "git_sha": "artifacts/2026-10-22/683675_formal_baseline/git_sha.txt",
            },
            "improved": {
                "config": "artifacts/2026-10-22/683676_formal_improved/experiment_config.json",
                "training_summary": "artifacts/2026-10-22/683676_formal_improved/training_summary.json",
                "metrics": "artifacts/2026-10-22/683676_formal_improved/test_metrics.json",
                "git_sha": "artifacts/2026-10-22/683676_formal_improved/git_sha.txt",
            },
        },
        {
            "group_id": "wide20_supplementary_matched",
            "selection_metric": "prostate_dice",
            "job_ids": {"baseline": "687753", "improved": "687754"},
            "baseline": {
                "config": "artifacts/2026-10-23/687753_formal_baseline_new/training_baseline/experiment_config.json",
                "training_summary": "artifacts/2026-10-23/687753_formal_baseline_new/training_baseline/training_summary.json",
                "metrics": "artifacts/2026-10-23/687753_formal_baseline_new/test_baseline/test_metrics.json",
                "git_sha": "artifacts/2026-10-23/687753_formal_baseline_new/training_code_git_sha.txt",
            },
            "improved": {
                "config": "artifacts/2026-10-23/687754_formal_improved_wide20_recovery/training_improved/experiment_config.json",
                "training_summary": "artifacts/2026-10-23/687754_formal_improved_wide20_recovery/training_improved/training_summary.json",
                "metrics": "artifacts/2026-10-23/687754_formal_improved_wide20_recovery/test_improved/test_metrics.json",
                "git_sha": "artifacts/2026-10-23/687754_formal_improved_wide20_recovery/training_code_git_sha.txt",
            },
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root containing ignored artifacts")
    parser.add_argument("--output", type=Path, default=Path("docs/results_summary.json"))
    args = parser.parse_args()
    groups = _default_groups()
    ensure_output_is_not_source(args.root, args.output, groups)
    summary = build_report_summary(args.root, groups)
    output = args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
