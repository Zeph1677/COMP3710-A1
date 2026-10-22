"""Validate frozen wide20 candidate evidence and final Slurm accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


_PROTOCOL = {
    "model": "improved",
    "base_channels": 20,
    "learning_rate": 0.001,
    "prostate_ce_weight": 4.0,
    "prostate_dice_weight": 4.0,
    "foreground_patch_probability": 0.5,
    "selection_metric": "prostate_dice",
    "seed": 42,
    "prostate_label": 5,
    "target_shape": [128, 128, 128],
    "train_patch_shape": [96, 96, 96],
}

_SACCT_FIELDS = ("JobID", "JobName", "State", "ExitCode", "Elapsed", "AllocTRES")


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_wide20_checkpoint(
    run_dir: Path,
    *,
    expected_git_sha: str,
    expected_manifest_sha256: str,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    """Return a recovery report only for a complete, protocol-compatible wide20 run."""

    training_dir = run_dir / "training_improved"
    checkpoint = training_dir / "best.pt"
    config = _load_json(training_dir / "experiment_config.json")
    summary = _load_json(training_dir / "training_summary.json")
    history = json.loads((training_dir / "history.json").read_text(encoding="utf-8"))
    if not isinstance(history, list):
        raise ValueError("history.json must contain a JSON array")
    actual_git_sha = (run_dir / "git_sha.txt").read_text(encoding="utf-8").strip()
    if actual_git_sha != expected_git_sha:
        raise ValueError(f"git_sha mismatch: {actual_git_sha}")
    if config.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("manifest_sha256 mismatch")
    for field, expected in _PROTOCOL.items():
        if config.get(field) != expected:
            raise ValueError(f"{field} mismatch")
    if summary.get("selection_metric") != "prostate_dice":
        raise ValueError("selection_metric mismatch")
    if summary.get("selected_epoch") != 11:
        raise ValueError("selected_epoch mismatch")
    if not any(isinstance(row, dict) and row.get("epoch") == 11 for row in history):
        raise ValueError("history does not contain selected_epoch")
    digest = _sha256(checkpoint)
    if expected_checkpoint_sha256 is not None and digest != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    return {
        "route": "reuse_frozen_checkpoint",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "git_sha": actual_git_sha,
        "manifest_sha256": expected_manifest_sha256,
        "selected_epoch": 11,
        "selection_metric": "prostate_dice",
    }


def validate_final_sacct(text: str, *, job_id: str) -> dict[str, str]:
    """Accept only the top-level completed, zero-exit sacct record for ``job_id``."""

    for raw_line in text.splitlines():
        columns = raw_line.split("|")
        if len(columns) != len(_SACCT_FIELDS):
            continue
        record = dict(zip(_SACCT_FIELDS, (value.strip() for value in columns)))
        if record["JobID"] != job_id:
            continue
        if record["State"] != "COMPLETED" or record["ExitCode"] != "0:0":
            raise ValueError("final accounting must be COMPLETED with ExitCode 0:0")
        return record
    raise ValueError(f"final accounting has no top-level row for job {job_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verify_wide20", type=Path, nargs="?", help="candidate_05_wide20 directory")
    parser.add_argument("--git-sha")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--sacct", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sacct:
        if not args.job_id:
            parser.error("--job-id is required with --sacct")
        result: dict[str, object] = validate_final_sacct(args.sacct.read_text(encoding="utf-8"), job_id=args.job_id)
    else:
        if not args.verify_wide20 or not args.git_sha or not args.manifest_sha256:
            parser.error("verify-wide20, --git-sha and --manifest-sha256 are required for checkpoint recovery")
        result = verify_wide20_checkpoint(args.verify_wide20, expected_git_sha=args.git_sha, expected_manifest_sha256=args.manifest_sha256, expected_checkpoint_sha256=args.checkpoint_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
