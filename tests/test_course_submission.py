from pathlib import Path

import pytest

from build_course_submission import build_submission, validate_submission


def _write(source: Path, relative: str, content: str = "content\n") -> None:
    path = source / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_submission_copies_only_declared_reproducibility_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "submission"
    required = (
        "README.md",
        "LICENSE.txt",
        "requirements.txt",
        ".gitignore",
        ".gitattributes",
        "src/train.py",
        "tests/test_train.py",
        "evidence/dataset_manifest.json",
        "evidence/sources/csiro-brief-data-description.txt",
        "docs/assets/model_architecture.drawio",
        "docs/assets/model_architecture.drawio.png",
    )
    for relative in required:
        _write(source, relative)
    _write(source, "data/HipMRI_Study_open/semantic_MRs/B006_Week0_LFOV.nii.gz")
    _write(source, "checkpoints/best.pt")
    _write(source, "local/archive/reference/README.md")

    copied = build_submission(source, output, required)

    assert copied == list(required)
    assert all((output / relative).is_file() for relative in required)
    assert not (output / "data").exists()
    assert not (output / "checkpoints").exists()
    assert not (output / "local").exists()
    assert validate_submission(output, required) == []


def test_validate_submission_reports_missing_and_forbidden_files(tmp_path: Path) -> None:
    output = tmp_path / "submission"
    _write(output, "README.md")
    _write(output, "weights/model.pt")

    errors = validate_submission(output, ("README.md", "src/train.py"))

    assert "missing required file: src/train.py" in errors
    assert "forbidden file: weights/model.pt" in errors


def test_build_submission_refuses_to_replace_nonempty_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "submission"
    _write(source, "README.md")
    _write(output, "keep.txt", "user data\n")

    with pytest.raises(FileExistsError, match="not empty"):
        build_submission(source, output, ("README.md",))

    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data\n"


def test_validate_submission_reports_broken_relative_readme_link(tmp_path: Path) -> None:
    output = tmp_path / "submission"
    _write(output, "README.md", "[manifest](evidence/dataset_manifest.json)\n")

    errors = validate_submission(output, ("README.md",))

    assert "broken README link: evidence/dataset_manifest.json" in errors


def test_validate_submission_accepts_existing_relative_and_remote_links(tmp_path: Path) -> None:
    output = tmp_path / "submission"
    _write(
        output,
        "README.md",
        "[manifest](evidence/dataset_manifest.json) [CSIRO](https://data.csiro.au/)\n",
    )
    _write(output, "evidence/dataset_manifest.json", "{}\n")

    assert validate_submission(output, ("README.md", "evidence/dataset_manifest.json")) == []
