"""Rank Improved-only validation sweep candidates without reading test artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MINIMUM_MEAN_GAIN = 0.01
MAXIMUM_WORST_PATIENT_DROP = 0.02
MAXIMUM_VRAM_MULTIPLIER = 1.35


def rank_candidates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Apply the registered validation-only gates and return deterministic ranking."""

    control = next((row for row in rows if row["slug"] == "control"), None)
    if control is None:
        raise ValueError("sweep requires exactly one control candidate")
    control_mean = float(control["validation_patient_prostate_dice"])
    control_worst = float(control["validation_worst_patient_prostate_dice"])
    control_vram = float(control["peak_train_vram_mb"])
    ranked: list[dict[str, object]] = []
    for row in rows:
        candidate = dict(row)
        mean_gain = float(candidate["validation_patient_prostate_dice"]) - control_mean
        worst_drop = control_worst - float(candidate["validation_worst_patient_prostate_dice"])
        vram_ratio = float(candidate["peak_train_vram_mb"]) / control_vram if control_vram else float("inf")
        accepted = candidate["slug"] != "control" and mean_gain >= MINIMUM_MEAN_GAIN and worst_drop <= MAXIMUM_WORST_PATIENT_DROP and vram_ratio <= MAXIMUM_VRAM_MULTIPLIER
        candidate.update({"mean_gain_over_control": mean_gain, "worst_patient_drop_from_control": worst_drop, "vram_ratio_to_control": vram_ratio, "accepted": accepted})
        ranked.append(candidate)
    return sorted(ranked, key=lambda row: (not bool(row["accepted"]), row["slug"] == "control", int(row["candidate_id"]) if not row["accepted"] else 0, -float(row["validation_patient_prostate_dice"]), -float(row["validation_worst_patient_prostate_dice"]), int(row["validation_prostate_false_positive_voxels"]) + int(row["validation_prostate_false_negative_voxels"])))


def _candidate_row(candidate_dir: Path) -> dict[str, object]:
    config = json.loads((candidate_dir / "training_improved" / "experiment_config.json").read_text(encoding="utf-8"))
    summary = json.loads((candidate_dir / "training_improved" / "training_summary.json").read_text(encoding="utf-8"))
    selected_epoch = float(summary["selected_epoch"])
    selected = next(row for row in summary["history"] if float(row["epoch"]) == selected_epoch)
    return {
        "candidate_id": int(candidate_dir.name.split("_", 2)[1]),
        "slug": candidate_dir.name.split("_", 2)[2],
        "learning_rate": config["learning_rate"],
        "base_channels": config["base_channels"],
        "prostate_ce_weight": config["prostate_ce_weight"],
        "prostate_dice_weight": config["prostate_dice_weight"],
        "foreground_patch_probability": config["foreground_patch_probability"],
        "selected_epoch": selected_epoch,
        "validation_patient_prostate_dice": selected["validation_patient_prostate_dice"],
        "validation_worst_patient_prostate_dice": selected["validation_worst_patient_prostate_dice"],
        "validation_prostate_false_positive_voxels": selected["validation_prostate_false_positive_voxels"],
        "validation_prostate_false_negative_voxels": selected["validation_prostate_false_negative_voxels"],
        "peak_train_vram_mb": summary["peak_train_vram_mb"],
        "training_seconds": summary["training_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    candidates = sorted(path for path in args.run_root.glob("candidate_*") if path.is_dir())
    if len(candidates) != 6:
        raise ValueError(f"expected six candidate directories, found {len(candidates)}")
    result = {
        "selection_boundary": "validation-only; no held-out test predictions were read",
        "rules": {"minimum_mean_gain": MINIMUM_MEAN_GAIN, "maximum_worst_patient_drop": MAXIMUM_WORST_PATIENT_DROP, "maximum_vram_multiplier": MAXIMUM_VRAM_MULTIPLIER},
        "candidates": rank_candidates([_candidate_row(path) for path in candidates]),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
