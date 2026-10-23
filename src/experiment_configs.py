"""Load the versioned, explicit HipMRI preflight and formal candidates."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _MODULE_DIR.parent if _MODULE_DIR.name == "src" else _MODULE_DIR
CONFIG_PATH = _PROJECT_DIR / "configs" / "hipmri_experiment_candidates.json"
_REQUIRED_FIELDS = {
    "seed", "target_shape", "train_patch_shape",
    "foreground_patch_probability", "prostate_ce_weight", "prostate_dice_weight",
}


def load_candidate_configuration(name: str, path: Path = CONFIG_PATH) -> dict[str, object]:
    """Return a validated immutable training candidate by its declared name."""

    document = json.loads(path.read_text(encoding="utf-8"))
    candidate = document.get("candidates", {}).get(name)
    if not isinstance(candidate, dict):
        raise ValueError(f"unknown HipMRI candidate configuration: {name}")
    missing = _REQUIRED_FIELDS - set(candidate)
    if missing:
        raise ValueError(f"candidate {name!r} is missing fields: {', '.join(sorted(missing))}")
    if candidate["seed"] != 42:
        raise ValueError("candidate must preserve seed 42")
    for field in ("target_shape", "train_patch_shape"):
        shape = candidate[field]
        if not isinstance(shape, list) or len(shape) != 3 or any(not isinstance(size, int) or size <= 0 or size % 8 for size in shape):
            raise ValueError(f"candidate {name!r} has invalid {field}")
    if any(patch > target for patch, target in zip(candidate["train_patch_shape"], candidate["target_shape"])):
        raise ValueError("train patch shape cannot exceed target shape")
    if not 0.0 <= float(candidate["foreground_patch_probability"]) <= 1.0:
        raise ValueError("foreground patch probability must be in [0, 1]")
    if float(candidate["prostate_ce_weight"]) <= 0 or float(candidate["prostate_dice_weight"]) <= 0:
        raise ValueError("prostate loss weights must be positive")
    return dict(candidate)


def shell_assignments(candidate: dict[str, object]) -> str:
    """Return safely quoted Bash assignments used by the two Slurm entrypoints."""

    values = {
        "CONFIG_SEED": candidate["seed"],
        "CONFIG_TARGET_SHAPE": " ".join(str(value) for value in candidate["target_shape"]),
        "CONFIG_TRAIN_PATCH_SHAPE": " ".join(str(value) for value in candidate["train_patch_shape"]),
        "CONFIG_FOREGROUND_PATCH_PROBABILITY": candidate["foreground_patch_probability"],
        "CONFIG_PROSTATE_CE_WEIGHT": candidate["prostate_ce_weight"],
        "CONFIG_PROSTATE_DICE_WEIGHT": candidate["prostate_dice_weight"],
    }
    return "\n".join(f"{name}={shlex.quote(str(value))}" for name, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    candidate = load_candidate_configuration(args.candidate)
    print(shell_assignments(candidate) if args.shell else json.dumps(candidate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
