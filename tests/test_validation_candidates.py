from pathlib import Path
import pytest
from validation_candidates import SPECS, candidate_arguments
from train import create_argument_parser


@pytest.mark.parametrize('name', ['P1', 'P2', 'P3'])
def test_registered_candidate_protocol_and_recorded_command(name, tmp_path):
    args = candidate_arguments(name, tmp_path/'run', tmp_path/'data', Path('evidence/dataset_manifest.json'))
    parsed = create_argument_parser().parse_args(args.recorded_command[2:])
    for key, value in SPECS[name].items():
        assert getattr(args, key) == getattr(parsed, key) == value
    assert args.architecture_version == 'improved_context_v2'
    assert args.batch_size == 1 and args.base_channels == 16 and args.seed == 42
    assert args.augmentation and args.selection_metric == 'prostate_dice'
    assert args.prostate_ce_weight == args.prostate_dice_weight == 4
    assert args.target_shape == [128]*3 and args.train_patch_shape == [96]*3


def test_dependent_candidates_cannot_be_started(tmp_path):
    for name in ('P4', 'A1', 'A2', 'L1'):
        with pytest.raises(ValueError):
            candidate_arguments(name, tmp_path, tmp_path, tmp_path)
