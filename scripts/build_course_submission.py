"""Export a shallow course contribution without raw data or model artifacts.

The export is a local preview. Transfer the reviewed changes using the existing
development history in the course fork; this command does not submit a PR.
"""

from pathlib import Path
import argparse
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true", help="Refresh this generated export beneath local/ only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if args.refresh and not output.is_relative_to(root / "local"):
        raise ValueError("Refresh is restricted to generated exports beneath local/")
    if output.exists() and not args.refresh:
        raise FileExistsError(f"Refusing to overwrite export: {output}")
    destination = output / "recognition" / "HipMRI_3D_UNet_s4910970"
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    for name in filter(None, paths):
        path = Path(name)
        if name.startswith("docs/superpowers/") or path.suffix.lower() == ".pdf":
            continue
        if path.parts[0] not in {"src", "tests", "scripts", "configs", "docs", "evidence"} and name not in {"README.md", "requirements.txt", "pyproject.toml", ".gitignore", ".gitattributes", "LICENSE.txt"}:
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
            content = source.read_text(encoding="utf-8").replace("src/", "").replace("src\\", "")
            if name == "pyproject.toml":
                content = content.replace('pythonpath = ["src"]', 'pythonpath = ["."]')
            if name == "tests/test_rangpur_smoke.py":
                for module in ("audit", "train", "predict"):
                    content = content.replace(f'assert "python {module}.py" not in script', f'assert "python src/{module}.py" not in script')
            target.write_text(content, encoding="utf-8", newline="\n")
        else:
            shutil.copy2(source, target)
    print(destination)


if __name__ == "__main__":
    main()
