"""Generate small, code-defined README assets without changing raw evidence."""

from __future__ import annotations

from pathlib import Path
import shutil
import tarfile

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def copy_asset(source: Path | str, destination: Path | str) -> None:
    """Copy an existing evidence asset byte-for-byte into the README asset folder."""

    origin, output = Path(source), Path(destination)
    if not origin.is_file():
        raise FileNotFoundError(f"evidence asset not found: {origin}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin, output)


def copy_tar_member(archive_path: Path | str, member_name: str, destination: Path | str) -> None:
    """Copy one named evidence member from a tar archive without extracting the archive."""

    archive, output = Path(archive_path), Path(destination)
    with tarfile.open(archive, "r:*") as bundle:
        member = bundle.getmember(member_name)
        source = bundle.extractfile(member)
        if source is None:
            raise FileNotFoundError(f"archive member is not a file: {member_name}")
        content = source.read()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)


def _box(axis: plt.Axes, x: float, y: float, label: str, colour: str) -> None:
    axis.add_patch(FancyBboxPatch((x, y), 1.5, 0.72, boxstyle="round,pad=0.04", facecolor=colour, edgecolor="#233044", linewidth=1.2))
    axis.text(x + 0.75, y + 0.36, label, ha="center", va="center", fontsize=8, wrap=True)


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.1, color="#425466"))


def render_model_architecture(output_path: Path | str) -> None:
    """Draw the implemented shared 3D U-Net skeleton and its explicit variants."""

    figure, axes = plt.subplots(2, 1, figsize=(13, 7.3), constrained_layout=True)
    variants = (
        ("Baseline 3D U-Net", "Conv block", "Final 1x1x1 logits only", "#DCEEFF"),
        ("Improved 3D U-Net", "Residual block", "Final 1x1x1 logits\ninference only", "#E7F7E5"),
    )
    for axis, (title, block, output_label, colour) in zip(axes, variants, strict=True):
        axis.set(xlim=(0, 14.7), ylim=(0, 3.2), title=title)
        axis.axis("off")
        entries = (
            (0.2, 1.25, "Input\n[B,1,D,H,W]"),
            (2.0, 2.1, f"Encoder 1\n{block}"),
            (3.8, 1.55, f"Encoder 2\n{block} + downsample"),
            (5.6, 1.0, f"Encoder 3\n{block} + downsample"),
            (7.4, 1.55, f"Bottleneck\n{block}"),
            (9.2, 1.0, "Decoder 3\nupsample + concat"),
            (11.0, 1.55, "Decoder 2\nupsample + concat"),
        )
        for x, y, label in entries:
            _box(axis, x, y, label, colour)
        _box(axis, 11.0, 0.12, "Decoder 1\nupsample + concat", colour)
        _box(axis, 12.75, 0.85, output_label, "#FFF2CC")
        for start, end in zip(((1.7, 1.61), (3.5, 2.46), (5.3, 1.91), (7.1, 1.36), (8.9, 1.91), (10.7, 1.36), (12.5, 1.91)), ((2.0, 2.46), (3.8, 1.91), (5.6, 1.36), (7.4, 1.91), (9.2, 1.36), (11.0, 1.91), (12.75, 1.21)), strict=True):
            _arrow(axis, start, end)
        for x1, y1, x2, y2 in ((2.75, 2.1, 11.0, 2.27), (4.55, 1.55, 9.2, 1.72), (6.35, 1.0, 11.0, 0.84)):
            _arrow(axis, (x1, y1), (x2, y2))
        if title.startswith("Improved"):
            _box(axis, 9.25, 2.5, "Auxiliary head 1\ntraining only", "#FCE4D6")
            _box(axis, 11.05, 2.5, "Auxiliary head 2\ntraining only", "#FCE4D6")
            _arrow(axis, (9.95, 1.72), (10.0, 2.5))
            _arrow(axis, (11.75, 2.27), (11.8, 2.5))
        axis.text(6.65, 0.02, "Three downsampling stages, bottleneck, three decoder stages, and skip concatenations", ha="center", va="bottom", fontsize=8, color="#425466")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
