from pathlib import Path

import tarfile

from documentation_assets import copy_asset, copy_tar_member, render_model_architecture


def test_render_model_architecture_writes_nonempty_png(tmp_path: Path):
    output = tmp_path / "architecture.png"

    render_model_architecture(output)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_copy_asset_preserves_existing_evidence_bytes(tmp_path: Path):
    source = tmp_path / "source.png"
    destination = tmp_path / "docs" / "failure.png"
    source.write_bytes(b"retained-panel-bytes")

    copy_asset(source, destination)

    assert destination.read_bytes() == b"retained-panel-bytes"


def test_copy_tar_member_preserves_archived_evidence_bytes(tmp_path: Path):
    archive = tmp_path / "evidence.tar.gz"
    source = tmp_path / "panel.png"
    source.write_bytes(b"archived-panel-bytes")
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname="comparison_panels/B037.png")
    destination = tmp_path / "docs" / "failure.png"

    copy_tar_member(archive, "comparison_panels/B037.png", destination)

    assert destination.read_bytes() == b"archived-panel-bytes"
