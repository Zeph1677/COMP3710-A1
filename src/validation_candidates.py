"""Registered P1/P2/P3 validation experiments; no held-out inference entry point."""
import argparse
import json
from pathlib import Path
from validation_completion import (audit_existing, make_training_arguments, run_record,
                                   sha256, write_json)
from validation_ablation_report import validation_gate
from train import run_training

SPECS = {
    'P1': {'epochs': 20, 'learning_rate': .0004, 'scheduler': 'cosine', 'eta_min': .0001},
    'P2': {'epochs': 30, 'learning_rate': .001, 'scheduler': 'none', 'eta_min': .00001},
    'P3': {'epochs': 30, 'learning_rate': .0004, 'scheduler': 'cosine', 'eta_min': .0001},
}


def candidate_arguments(name, output, data_root, manifest):
    if name not in SPECS:
        raise ValueError('only registered P1/P2/P3 are executable')
    args = make_training_arguments('improved', True, 16, 42, output, data_root, manifest)
    for key, value in SPECS[name].items():
        setattr(args, key, value)
        flag = '--' + key.replace('_', '-')
        if flag in args.recorded_command:
            args.recorded_command[args.recorded_command.index(flag)+1] = str(value)
        else:
            args.recorded_command.extend([flag, str(value)])
    return args


def run_candidate(args):
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError('refusing to overwrite candidate output')
    audit = audit_existing(args.existing_root, args.manifest, args.existing_accounting)
    output.mkdir(parents=True)
    write_json(output/'existing_evidence_audit.json', audit)
    training = candidate_arguments(args.candidate, output/'training', args.data_root, args.manifest)
    write_json(output/'registered_candidate.json', {
        'candidate': args.candidate, 'specification': SPECS[args.candidate],
        'command': training.recorded_command, 'manifest_sha256': sha256(args.manifest),
        'architecture_version': 'improved_context_v2', 'initialization': 'fresh seed 42; no resume',
        'precision': 'float32', 'automatic_held_out_evaluation': False,
        'training_update_accounting': 'batch=1, one optimizer update per training volume per epoch; no accumulation',
    })
    run_training(training)
    record = run_record(output/'training', expected_epochs=training.epochs)
    write_json(output/'verified_record.json', record)
    baseline, reference = audit['reused_arms']['D'], audit['reused_arms']['B']
    scores = record['selected']['validation_patient_prostate_dice_by_patient']
    comparisons = {}
    for label, other in [('B', reference), ('D', baseline)]:
        old = other['selected']['validation_patient_prostate_dice_by_patient']
        comparisons[label] = {'mean_difference': record['mean']-other['mean'],
                              'patient_differences': {p: scores[p]-old[p] for p in sorted(scores)}}
    write_json(output/'candidate_report.json', {
        'candidate': args.candidate, 'status': 'validation_complete',
        'comparisons': comparisons, 'engineering_gate_against_D': validation_gate(record, baseline),
        'mean_exceeds_B': record['mean'] > reference['mean'], 'held_out_test_run': False,
        'resource_caveat': 'historical B/D hardware measurements are not a new uniform paired benchmark',
        'time_definition': 'entire validation pass, not per-volume inference latency',
        'statistical_scope': 'single seed exploration; no stable improvement claim',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=list(SPECS), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('evidence/dataset_manifest.json'))
    parser.add_argument('--existing-root', type=Path, required=True)
    parser.add_argument('--existing-accounting', type=Path, required=True)
    run_candidate(parser.parse_args())
