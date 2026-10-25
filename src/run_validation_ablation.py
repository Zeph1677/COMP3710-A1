"""Run registered A/B/C/D validation ablations without invoking held-out inference."""

import argparse
import hashlib
import json
from pathlib import Path
from train import create_argument_parser, run_training

ARMS = {
    "A": ("improved", False, "none"),
    "B": ("improved", True, "none"),
    "C": ("improved", True, "cosine"),
    "D": ("baseline", False, "none"),
}


def main():
    """Run requested arms in CLI order and write validation-only summaries.

    The recorded argument lists are the actual inputs passed to training; no
    held-out evaluation is scheduled, even when all validation gates pass.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for arm in args.arms:
        model, augmentation, scheduler = ARMS[arm]
        command = [
            "--manifest",
            "evidence/dataset_manifest.json",
            "--data-root",
            str(args.data_root),
            "--output-dir",
            str(args.output_root / arm),
            "--model",
            model,
            "--epochs",
            "20",
            "--batch-size",
            "1",
            "--learning-rate",
            "0.001",
            "--base-channels",
            "16",
            "--target-shape",
            "128",
            "128",
            "128",
            "--train-patch-shape",
            "96",
            "96",
            "96",
            "--foreground-patch-probability",
            "0.5",
            "--seed",
            "42",
            "--device",
            args.device,
            "--prostate-label",
            "5",
            "--prostate-label-source",
            "CSIRO DOI 10.25919/45T8-P065, BriefDataDescription.txt Labels: 5=Prostate",
            "--prostate-ce-weight",
            "4",
            "--prostate-dice-weight",
            "4",
            "--selection-metric",
            "prostate_dice",
            "--scheduler",
            scheduler,
            "--eta-min",
            "0.00001",
            "--validation-gifs",
        ]
        if augmentation:
            command.append("--augmentation")
        (args.output_root / f"{arm}_command.json").write_text(
            json.dumps(command, indent=2) + "\n"
        )
        summary = run_training(create_argument_parser().parse_args(command))
        results.append(
            {
                "arm": arm,
                "selected_epoch": summary["selected_epoch"],
                "validation_patient_prostate_dice": summary["selection_value"],
            }
        )
        (args.output_root / "completed_arms.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
    if all(
        (args.output_root / arm / "training_summary.json").is_file() for arm in ARMS
    ):
        from validation_ablation_report import summarize

        summarize(args.output_root)
    # This runner deliberately never opens test predictions or launches a test command.


if __name__ == "__main__":
    main()
