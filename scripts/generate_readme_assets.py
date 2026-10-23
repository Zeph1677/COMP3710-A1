"""Generate tracked README figures from code and retained main-experiment evidence."""

from __future__ import annotations

import argparse
import json
import tarfile
import torch
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src" if (project_root / "src").is_dir() else project_root))

from documentation_assets import render_model_architecture
from compare_failures import render_comparative_prostate_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--paired-comparison-archive", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, default=Path("artifacts/final_rangpur_evidence_20261023/predictions"))
    args = parser.parse_args()
    render_model_architecture(args.output_dir / "model_architecture.png")
    with tarfile.open(args.paired_comparison_archive) as archive:
        member = archive.extractfile("artifacts/paired_failure_comparison_683675_683676/selection_manifest.json")
        if member is None:
            raise ValueError("Missing historical case selection manifest")
        selection = json.load(member)
    for entry in selection["selected_cases"]:
        case = entry["case_id"]
        baseline = torch.load(args.prediction_root / "683675_formal_baseline" / "predictions" / f"{case}.pt", map_location="cpu", weights_only=False)
        improved = torch.load(args.prediction_root / "683676_formal_improved" / "predictions" / f"{case}.pt", map_location="cpu", weights_only=False)
        if not torch.equal(baseline["target"], improved["target"]):
            raise ValueError(f"Different reference masks: {case}")
        destination = args.output_dir / ("failure_case_b037.png" if case == "B037_Week0" else f"failure_case_{case.lower()}.png")
        index = render_comparative_prostate_panel(baseline["image"], baseline["target"], baseline["prediction"], improved["prediction"], int(baseline["prostate_label"]), destination)
        if index != entry["slice_index"]:
            raise ValueError(f"Historical slice selection changed: {case}")
    (args.output_dir / "failure_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
