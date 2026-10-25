"""Exercise package boundaries with retained bytes and an isolated Git index."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def load_script(name):
    path = REPOSITORY / "scripts" / name
    if not path.is_file():
        pytest.skip("Local packaging helper is not part of the course export")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_flattens_source_and_limits_distributed_evidence(tmp_path, monkeypatch):
    module = load_script("build_course_submission.py")
    repository = tmp_path / "repository"
    repository.mkdir()
    names = [
        "src/train.py",
        "tests/test_train.py",
        "pyproject.toml",
        "docs/EXPERIMENT_EVIDENCE.md",
        "evidence/dataset_manifest.json",
        "docs/assets/context_v2_ablation_692354/best_ABCD_preview.png",
        "docs/assets/context_v2_ablation_692354/failure_M030_Week0.png",
        "docs/assets/failure_case_b037.png",
        "docs/assets/context_v2_ablation_692354/other.png",
        "docs/assets/context_v2_ablation_692354/animation.gif",
        "docs/assets/gallery.HTML",
        "docs/assets/.$architecture.drawio.bkp",
        "docs/personal_reference.md",
        "docs/execution_plan.md",
        "docs/superpowers/specs/notes.md",
        "evidence/paper.PDF",
        "artifacts/raw.json",
        "evidence/artifacts/raw.json",
    ]
    for name in names:
        file = repository / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(
            'pythonpath = ["src"]\n' if name == "pyproject.toml" else "retained bytes\n"
        )
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "add", "-A"], cwd=repository, check=True)
    monkeypatch.setattr(
        module, "__file__", str(repository / "scripts" / "build_course_submission.py")
    )
    output = tmp_path / "export"
    monkeypatch.setattr(
        sys, "argv", ["build_course_submission.py", "--output", str(output)]
    )

    module.main()

    course = output / "recognition" / "HipMRI_3D_UNet_s4910970"
    actual = {
        p.relative_to(course).as_posix() for p in course.rglob("*") if p.is_file()
    }
    assert actual == {
        "train.py",
        "tests/test_train.py",
        "pyproject.toml",
        "docs/EXPERIMENT_EVIDENCE.md",
        "evidence/dataset_manifest.json",
        "docs/assets/context_v2_ablation_692354/best_ABCD_preview.png",
        "docs/assets/context_v2_ablation_692354/failure_M030_Week0.png",
        "docs/assets/failure_case_b037.png",
    }
    assert (course / "pyproject.toml").read_text() == 'pythonpath = ["."]\n'
    assert (course / "train.py").read_bytes() == b"retained bytes\n"
    sentinel = output / "keep.txt"
    sentinel.write_text("existing export")
    with pytest.raises(FileExistsError):
        module.main()
    assert sentinel.read_text() == "existing export"


def test_compact_publication_copies_only_two_panels_without_inference(
    tmp_path, monkeypatch
):
    module = load_script("publish_context_validation_assets.py")
    source = tmp_path / "full"
    source.mkdir()
    for name in [
        "best_ABCD_preview.png",
        "failure_M030_Week0.png",
        "animation.gif",
        "gallery.html",
    ]:
        (source / name).write_bytes(name.encode())
    destination = tmp_path / "compact"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_context_validation_assets.py",
            "--output",
            str(source),
            "--publish-compact",
            "--compact-output",
            str(destination),
        ],
    )
    monkeypatch.setattr(
        module.torch,
        "set_num_threads",
        lambda _: pytest.fail("Compact publication must not enter inference setup"),
    )

    module.main()

    assert {p.name for p in destination.iterdir()} == {
        "best_ABCD_preview.png",
        "failure_M030_Week0.png",
    }
    for path in destination.iterdir():
        assert path.read_bytes() == (source / path.name).read_bytes()


def test_compact_publication_validates_both_panels_before_writing(tmp_path):
    module = load_script("publish_context_validation_assets.py")
    source = tmp_path / "full"
    source.mkdir()
    (source / "best_ABCD_preview.png").write_bytes(b"preview")
    destination = tmp_path / "compact"
    with pytest.raises(FileNotFoundError):
        module.publish_compact(source, destination)
    assert not destination.exists()

def test_publication_cli_imports_in_flattened_course_layout(tmp_path):
    """A shallow course export has modules beside scripts/, with no src/ folder."""
    import shutil
    source = REPOSITORY / 'src'
    if not source.is_dir():
        source = REPOSITORY
    for path in source.glob('*.py'):
        shutil.copy2(path, tmp_path / path.name)
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copy2(REPOSITORY / 'scripts/publish_context_validation_assets.py', scripts)
    result = subprocess.run([sys.executable, str(scripts / 'publish_context_validation_assets.py'), '--help'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--publish-compact' in result.stdout

def test_export_preserves_registered_context_gifs_and_gallery():
    module = load_script('build_course_submission.py')
    for name in ['B_progress.gif', 'best_ABCD.gif', 'A_validation_epoch_1.gif', 'gallery.html', 'asset_provenance.json']:
        assert module.should_export(Path('docs/assets/context_v2_ablation_692354') / name), name
    assert not module.should_export(Path('docs/assets/context_v2_ablation_692354/unknown.gif'))
    assert not module.should_export(Path('artifacts/raw.pt'))
