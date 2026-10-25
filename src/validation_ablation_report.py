"""Summarize complete matched validation ablations; never automatically test."""

import argparse
import hashlib
import json
from pathlib import Path
from plot_training import render_training_curves, render_learning_rate_curve

# Operational screening criteria, declared before B/C results; not course thresholds.
RULES = {
    "minimum_mean_gain": 0.01,
    "maximum_worst_drop": 0.02,
    "maximum_fn_ratio": 1.10,
    "maximum_vram_ratio": 1.35,
    "maximum_inference_time_ratio": 2.0,
}


def validation_gate(candidate, baseline):
    """Return declared validation/resource checks, never test authorization."""
    checks = {
        "mean_gain": candidate["mean"] - baseline["mean"] >= RULES["minimum_mean_gain"],
        "worst_patient": candidate["worst"]
        >= baseline["worst"] - RULES["maximum_worst_drop"],
        "fn": candidate["fn"] <= baseline["fn"] * RULES["maximum_fn_ratio"],
        "vram": baseline["vram"] > 0
        and candidate["vram"] <= baseline["vram"] * RULES["maximum_vram_ratio"],
        "inference_time": baseline["time"] > 0
        and candidate["time"]
        <= baseline["time"] * RULES["maximum_inference_time_ratio"],
    }
    return {
        "checks": checks,
        "eligible_for_test_review": all(checks.values()),
        "automatic_test_permitted": False,
    }


def summarize(root):
    """Read matched A/B/C/D runs, write validation reports and return their data.

    Inputs are retained training summaries/configs and checkpoint bytes under
    ``root``. No dataset or test predictions are opened by this report.
    """
    root = Path(root)
    runs = {}
    protocol_fields = [
        "manifest_sha256",
        "seed",
        "target_shape",
        "train_patch_shape",
        "base_channels",
        "epochs",
        "batch_size",
        "learning_rate",
        "foreground_patch_probability",
        "prostate_label",
        "prostate_ce_weight",
        "prostate_dice_weight",
        "selection_metric",
        "device",
    ]
    for arm in "ABCD":
        directory = root / arm
        config = json.loads((directory / "experiment_config.json").read_text())
        summary = json.loads((directory / "training_summary.json").read_text())
        if len(summary["history"]) != config["epochs"]:
            raise ValueError("incomplete training history")
        row = next(
            r for r in summary["history"] if r["epoch"] == summary["selected_epoch"]
        )
        runs[arm] = {
            "config": config,
            "summary": summary,
            "selected": row,
            "mean": row["validation_patient_prostate_dice"],
            "worst": row["validation_worst_patient_prostate_dice"],
            "fn": row["validation_prostate_false_negative_voxels"],
            "fp": row["validation_prostate_false_positive_voxels"],
            "vram": summary["peak_train_vram_mb"],
            "time": row["validation_seconds"],
            "checkpoint_sha256": hashlib.sha256(
                (directory / "best.pt").read_bytes()
            ).hexdigest(),
        }
    for arm in "ABC":
        if any(
            runs[arm]["config"][field] != runs["D"]["config"][field]
            for field in protocol_fields
        ):
            raise ValueError("unmatched validation protocol")
    result = {
        "scope": "validation only; no held-out data read",
        "rules": RULES,
        "rules_status": "engineering screening criteria, not statistical significance or clinical assurance",
        "arms": {
            arm: {k: v for k, v in run.items() if k not in {"config", "summary"}}
            for arm, run in runs.items()
        },
        "gates": {arm: validation_gate(runs[arm], runs["D"]) for arm in "ABC"},
    }
    (root / "validation_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    histories = {arm: run["summary"]["history"] for arm, run in runs.items()}
    render_training_curves(
        histories, root / "training_loss_and_validation_dice.png", "prostate_dice"
    )
    render_learning_rate_curve(histories, root / "learning_rate_schedule.png")
    lines = [
        "# Context v2 A/B/C/D validation report",
        "",
        "Validation only. No held-out test was run by this workflow.",
        "",
        "| Arm | Selected epoch | Mean patient Dice | Worst patient Dice | FN | FP | Parameters | Peak train MiB | Training seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm, run in runs.items():
        row = run["selected"]
        summary = run["summary"]
        lines.append(
            f"| {arm} | {row['epoch']:g} | {run['mean']:.6f} | {run['worst']:.6f} | {run['fn']:g} | {run['fp']:g} | {summary['parameter_count']} | {run['vram']:.1f} | {summary['training_seconds']:.1f} |"
        )
    lines += [
        "",
        "## Screening against matched Baseline D",
        "",
        "Operational criteria are not statistical significance. A passing gate requires manual review of validation failure panels before any held-out evaluation.",
        "",
    ]
    for arm, gate in result["gates"].items():
        lines.append(
            f"- {arm}: eligible for test review = {gate['eligible_for_test_review']}; checks = {gate['checks']}"
        )
    lines += [
        "",
        "Checkpoint SHA-256 values are recorded in validation_comparison.json. Historical checkpoints and test artifacts were not overwritten.",
    ]
    (root / "validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run_root)


if __name__ == "__main__":
    main()
