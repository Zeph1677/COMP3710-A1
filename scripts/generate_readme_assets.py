"""Generate tracked README figures from code and retained main-experiment evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from documentation_assets import copy_tar_member, render_model_architecture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--paired-comparison-archive", type=Path, required=True)
    args = parser.parse_args()
    render_model_architecture(args.output_dir / "model_architecture.png")
    copy_tar_member(
        args.paired_comparison_archive,
        "artifacts/paired_failure_comparison_683675_683676/comparison_panels/B037__B037_Week0.png",
        args.output_dir / "failure_case_b037.png",
    )


if __name__ == "__main__":
    main()
