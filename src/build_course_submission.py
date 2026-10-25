"""Build and validate the minimal reproducible HipMRI course submission."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import unquote


CORE_FILES: tuple[str, ...] = (
    ".gitattributes",
    ".gitignore",
    "LICENSE.txt",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "src/audit.py",
    "src/build_course_submission.py",
    "src/dataset.py",
    "src/modules.py",
    "src/plot_training.py",
    "src/predict.py",
    "src/train.py",
    "src/validation_visuals.py",
    "tests/test_audit.py",
    "tests/test_course_submission.py",
    "tests/test_dataset.py",
    "tests/test_modules.py",
    "tests/test_plot_training.py",
    "tests/test_predict.py",
    "tests/test_train.py",
    "tests/test_training_extensions.py",
    "evidence/SOURCES.md",
    "evidence/dataset_manifest.json",
    "evidence/sources/csiro-brief-data-description.txt",
    "docs/assets/model_architecture.drawio",
    "docs/assets/model_architecture.drawio.png",
    "docs/requirements-verified-local.txt",
    "docs/results_summary.json",
)

FORBIDDEN_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "checkpoints",
    "data",
    "formal_results",
    "local",
    "temp",
}
FORBIDDEN_SUFFIXES = (".ckpt", ".nii", ".nii.gz", ".pt", ".pth")


def _normalise_files(files: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(Path(item).as_posix()) for item in files)
    if len(values) != len(set(values)):
        raise ValueError("required file list contains duplicates")
    for value in values:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"required file must be repository-relative: {value}")
    return values


def _is_forbidden(relative: Path) -> bool:
    lowered_parts = {part.lower() for part in relative.parts}
    name = relative.name.lower()
    return bool(lowered_parts & FORBIDDEN_DIRECTORY_NAMES) or any(
        name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES
    )


def validate_submission(output: Path | str, required_files: Iterable[str] = CORE_FILES) -> list[str]:
    """Return deterministic validation errors for an exported directory."""

    root = Path(output)
    required = _normalise_files(required_files)
    errors = [f"missing required file: {relative}" for relative in required if not (root / relative).is_file()]
    if root.exists():
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
            relative = path.relative_to(root)
            if _is_forbidden(relative):
                errors.append(f"forbidden file: {relative.as_posix()}")
        readme = root / "README.md"
        if readme.is_file():
            markdown = readme.read_text(encoding="utf-8")
            for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", markdown):
                target = match.group(1).strip().strip("<>").split("#", 1)[0]
                if not target or re.match(r"^(?:https?://|mailto:)", target, flags=re.IGNORECASE):
                    continue
                decoded = unquote(target)
                if not (root / decoded).exists():
                    errors.append(f"broken README link: {decoded}")
    return errors


def build_submission(
    source: Path | str,
    output: Path | str,
    required_files: Iterable[str] = CORE_FILES,
) -> list[str]:
    """Copy the declared files into a new or empty submission directory."""

    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    required = _normalise_files(required_files)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"submission output is not empty: {output_root}")
    missing = [relative for relative in required if not (source_root / relative).is_file()]
    if missing:
        raise FileNotFoundError("missing source files: " + ", ".join(missing))
    output_root.mkdir(parents=True, exist_ok=True)
    for relative in required:
        source_path = source_root / relative
        output_path = output_root / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)
    errors = validate_submission(output_root, required)
    if errors:
        raise ValueError("invalid submission:\n" + "\n".join(errors))
    return list(required)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    errors = validate_submission(args.output) if args.validate_only else []
    if not args.validate_only:
        copied = build_submission(args.source, args.output)
        print(f"Created {args.output.resolve()} with {len(copied)} files.")
        return 0
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"Validated {args.output.resolve()}: {len(CORE_FILES)} required files, no forbidden files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
