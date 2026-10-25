"""Export a shallow course contribution without raw data or model artifacts.

The export is a local preview. Transfer the reviewed changes using the existing
development history in the course fork; this command does not submit a PR.
"""

from pathlib import Path
import argparse
import shutil
import subprocess


DOCUMENTATION_PATHS = {
    "docs/superpowers/plans/EXPERIMENT_EVIDENCE.md",
    "docs/superpowers/plans/REFERENCE_LEARNING.md",
    "docs/superpowers/plans/EXPERIMENT_ENVIRONMENT.md",
    "docs/superpowers/plans/HIPMRI_3D_HARD_EXECUTION_FLOW.md",
    "docs/superpowers/specs/context_v2_ablation_results_20261025.md",
    "docs/requirements-verified-local.txt",
    "docs/results_summary.json",
}
# Exact whitelist preserves the reviewed animation gallery, never raw tensors.
CONTEXT_ASSETS = {
    "baseline_vs_improved_training.png",
    'A_validation_best_epoch.gif',
    'A_validation_epoch_1.gif',
    'A_validation_epoch_5.gif',
    'B_progress.gif',
    'B_progress_preview.png',
    'B_validation_best_epoch.gif',
    'B_validation_epoch_1.gif',
    'B_validation_epoch_5.gif',
    'C_validation_best_epoch.gif',
    'C_validation_epoch_1.gif',
    'C_validation_epoch_5.gif',
    'D_validation_best_epoch.gif',
    'D_validation_epoch_1.gif',
    'D_validation_epoch_5.gif',
    'asset_provenance.json',
    'best_ABCD.gif',
    'best_ABCD_preview.png',
    'failure_G021_Week6.png',
    'failure_M030_Week0.png',
    'failure_N010_Week3.png',
    'failure_S022_Week2.png',
    'failure_W012_Week0.png',
    'gallery.html',
    'learning_rate_schedule.png',
    'paired_validation_diagnostics.json',
    'patient_validation_dice.png',
    'training_loss_and_validation_dice.png',
}


def should_export(path: Path) -> bool:
    """Select distributable tracked inputs; raw runs and personal notes stay local."""
    if path.parts[:3] == ("docs", "assets", "context_v2_ablation_692354"):
        return len(path.parts) == 4 and path.name in CONTEXT_ASSETS
    if path.suffix.lower() in {".pdf", ".gif", ".html", ".htm", ".bkp", ".bak"}:
        return False
    if any(part.startswith(".$") or part.endswith("~") for part in path.parts):
        return False
    if any(
        part in {"artifacts", "__pycache__", "references", "reference"}
        for part in path.parts
    ):
        return False
    if path.parts[0] == "docs":
        if path.as_posix() in DOCUMENTATION_PATHS:
            return True
        if path.parts[1] != "assets":
            return False
        if "context_v2_ablation_692354" in path.parts:
            return len(path.parts) == 4 and path.name in CONTEXT_ASSETS
        return len(path.parts) == 3
    return path.parts[0] in {"src", "tests", "scripts", "configs", "evidence"} or str(
        path
    ) in {
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        ".gitignore",
        ".gitattributes",
        "LICENSE.txt",
    }


def main():
    """Flatten src/ into a fresh course directory, retaining the existing CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh this generated export beneath local/ only",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if args.refresh and not output.is_relative_to(root / "local"):
        raise ValueError("Refresh is restricted to generated exports beneath local/")
    if output.exists() and not args.refresh:
        raise FileExistsError(f"Refusing to overwrite export: {output}")
    destination = output / "recognition" / "HipMRI_3D_UNet_s4910970"
    paths = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    for name in filter(None, paths):
        path = Path(name)
        if not should_export(path):
            continue
        if name == "scripts/build_course_submission.py":
            continue
        source = root / path
        if not source.is_file():
            continue
        relative = Path(*path.parts[1:]) if path.parts[0] == "src" else path
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".md", ".py", ".slurm", ".toml"}:
            content = (
                source.read_text(encoding="utf-8")
                .replace("src/", "")
                .replace("src\\", "")
            )
            if name == "pyproject.toml":
                content = content.replace('pythonpath = ["src"]', 'pythonpath = ["."]')
            if name == "tests/test_rangpur_smoke.py":
                for module in ("audit", "train", "predict"):
                    content = content.replace(
                        f'assert "python {module}.py" not in script',
                        f'assert "python src/{module}.py" not in script',
                    )
            target.write_text(content, encoding="utf-8", newline="\n")
        else:
            shutil.copy2(source, target)
    print(destination)


if __name__ == "__main__":
    main()
