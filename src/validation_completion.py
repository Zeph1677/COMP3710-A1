"""Complete missing validation evidence and gated paired seeds; never evaluate held-out MRI."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics

import torch
from dataset import AugmentationConfig
from hipmri_resource_probe import profile_resources, select_resource_width
from predict import load_checkpoint_model
from train import create_argument_parser, load_audited_pairs, run_training, select_validation_checkpoint
from validation_ablation_report import RULES, validation_gate

PROSTATE_SOURCE = 'CSIRO DOI 10.25919/45T8-P065, BriefDataDescription.txt Labels: 5=Prostate'
EXPECTED_MANIFEST_SHA = '5457a611c5611f944a26d214a1f325b803dded66f70df5fda3fd009642751a07'
COMMON_FIELDS = ('manifest_sha256', 'seed', 'target_shape', 'train_patch_shape', 'base_channels', 'epochs', 'batch_size', 'learning_rate', 'foreground_patch_probability', 'prostate_label', 'prostate_ce_weight', 'prostate_dice_weight', 'selection_metric')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def validate_final_accounting(text):
    """Require the parent job's final success, not an EXIT trap's running snapshot."""
    matches = [line.strip().split('|') for line in text.lstrip('\ufeff').splitlines() if line.strip().startswith('692354|')]
    if len(matches) != 1 or len(matches[0]) < 5:
        raise ValueError('final parent-job accounting for 692354 is missing or duplicated')
    row = matches[0]
    if row[2] != 'COMPLETED' or row[3] != '0:0':
        raise ValueError('reused job must be COMPLETED with exit code 0:0')
    return {'job_id': row[0], 'job_name': row[1], 'state': row[2], 'exit_code': row[3], 'elapsed': row[4], 'raw': '|'.join(row)}


def run_record(directory, expected_epochs=20):
    """Read a complete run and independently verify checkpoint-selected identity."""
    directory = Path(directory)
    config = read_json(directory / 'experiment_config.json')
    history = read_json(directory / 'history.json')
    summary = read_json(directory / 'training_summary.json')
    if config['epochs'] != expected_epochs or [r['epoch'] for r in history] != list(range(1, expected_epochs + 1)):
        raise ValueError(f'incomplete {expected_epochs}-epoch history: {directory}')
    if history != summary['history']:
        raise ValueError('standalone history and summary differ')
    selected = select_validation_checkpoint(history, 'prostate_dice')
    if selected['epoch'] != summary['selected_epoch'] or selected['validation_patient_prostate_dice'] != summary['selection_value']:
        raise ValueError('summary does not identify the validation-selected epoch')
    checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    if checkpoint['epoch'] != int(selected['epoch']) or checkpoint['config'] != config:
        raise ValueError('checkpoint identity/config differs from selected run')
    # Loading weights checks all parameter keys and shapes, without reading any MRI.
    model, _ = load_checkpoint_model(directory / 'best.pt', torch.device('cpu'))
    count = sum(p.numel() for p in model.parameters())
    del model, checkpoint
    if count != summary['parameter_count']:
        raise ValueError('parameter count does not match the declared model')
    return {'directory': str(directory), 'config': config, 'selected': selected,
            'mean': selected['validation_patient_prostate_dice'],
            'worst': selected['validation_worst_patient_prostate_dice'],
            'fn': selected['validation_prostate_false_negative_voxels'],
            'fp': selected['validation_prostate_false_positive_voxels'],
            'vram': summary['peak_train_vram_mb'], 'time': selected['validation_seconds'],
            'training_seconds': summary['training_seconds'], 'parameter_count': count,
            'checkpoint_sha256': sha256(directory / 'best.pt'),
            'history_sha256': sha256(directory / 'history.json')}


def audit_existing(existing_root, manifest, accounting_path):
    """Freeze reused A/B/D using data metadata, strict weights and final accounting."""
    root = Path(existing_root)
    accounting = validate_final_accounting(Path(accounting_path).read_text(encoding='utf-8-sig'))
    digest = sha256(manifest)
    if digest != EXPECTED_MANIFEST_SHA:
        raise ValueError('the fixed patient manifest has changed')
    _, split_ids, _ = load_audited_pairs(manifest)
    if {k: len(v) for k, v in split_ids.items()} != {'train': 27, 'validation': 6, 'test': 5}:
        raise ValueError('unexpected patient split')
    if sha256(root.parent / 're_audited_manifest.json') != digest:
        raise ValueError('reused data audit differs from the frozen manifest')
    runs = {arm: run_record(root / arm) for arm in ('A', 'B', 'D')}
    expected = {'manifest_sha256': digest, 'seed': 42, 'target_shape': [128]*3,
                'train_patch_shape': [96]*3, 'base_channels': 16, 'epochs': 20,
                'batch_size': 1, 'learning_rate': .001, 'foreground_patch_probability': .5,
                'prostate_label': 5, 'prostate_ce_weight': 4., 'prostate_dice_weight': 4.,
                'selection_metric': 'prostate_dice'}
    for arm, record in runs.items():
        config = record['config']
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError(f'unmatched reused configuration: {arm}')
        if config.get('scheduler', 'none') != 'none':
            raise ValueError('reused arm must use fixed LR')
        if config.get('model') != ('baseline' if arm == 'D' else 'improved'):
            raise ValueError('reused model identity mismatch')
        if arm != 'D' and config.get('architecture_version') != 'improved_context_v2':
            raise ValueError('only Context v2 can be reused here')
        if config.get('validation_augmentation', False) or config.get('test_augmentation', False):
            raise ValueError('evaluation augmentation is forbidden')
        augmentation = asdict(AugmentationConfig()) if arm == 'B' else None
        if config.get('augmentation_config') != augmentation:
            raise ValueError('reused augmentation differs from the fixed factorial design')
        scores = record['selected']['validation_patient_prostate_dice_by_patient']
        if set(scores) != split_ids['validation']:
            raise ValueError('reused validation patients differ from manifest')
    return {'accounting': accounting, 'manifest_sha256': digest, 'reused_arms': runs,
            'scope': 'metadata and strict checkpoint loading only; no test MRI opened'}


def make_training_arguments(model, augmentation, width, seed, output, data_root, manifest):
    """Vary training RNG only; the fixed patient manifest is never rebuilt by seed."""
    command = ['--manifest', str(manifest), '--data-root', str(data_root), '--output-dir', str(output),
               '--model', model, '--epochs', '20', '--batch-size', '1', '--learning-rate', '.001',
               '--base-channels', str(width), '--target-shape', '128', '128', '128',
               '--train-patch-shape', '96', '96', '96', '--foreground-patch-probability', '.5',
               '--seed', str(seed), '--device', 'cuda', '--prostate-label', '5',
               '--prostate-label-source', PROSTATE_SOURCE, '--prostate-ce-weight', '4',
               '--prostate-dice-weight', '4', '--selection-metric', 'prostate_dice', '--scheduler', 'none',
               '--validation-gifs']
    if model == 'improved':
        command += ['--architecture-version', 'improved_context_v2', '--dropout-probability', '.3']
    if augmentation:
        command.append('--augmentation')
    args = create_argument_parser().parse_args(command)
    args.recorded_command = ['python', 'src/train.py', *command]
    return args


def summarize_seed_pairs(rows):
    rows = sorted(rows, key=lambda row: row['seed'])
    if [r['seed'] for r in rows] != [42, 43, 44]:
        raise ValueError('all three fixed paired seeds 42/43/44 are required')
    baseline = [r['baseline']['mean'] for r in rows]
    improved = [r['improved']['mean'] for r in rows]
    difference = [i-b for b, i in zip(baseline, improved)]
    return {'seeds': [42, 43, 44], 'baseline_mean': statistics.mean(baseline),
            'baseline_sample_sd': statistics.stdev(baseline), 'improved_mean': statistics.mean(improved),
            'improved_sample_sd': statistics.stdev(improved), 'paired_differences': difference,
            'paired_difference_mean': statistics.mean(difference),
            'paired_difference_sample_sd': statistics.stdev(difference),
            'all_paired_differences_positive': all(x > 0 for x in difference),
            'interpretation': 'three seeds estimate training variation; not proof of statistical significance'}


def train_one(root, label, model, width, seed, data_root, manifest):
    directory = root / label
    if directory.exists():
        raise FileExistsError(f'refusing to retrain or overwrite {directory}')
    args = make_training_arguments(model, True, width, seed, directory, data_root, manifest)
    write_json(root / f'{label}_command.json', args.recorded_command)
    print(f'Training {label}: model={model}, width={width}, seed={seed}', flush=True)
    run_training(args)
    record = run_record(directory)
    write_json(root / f'{label}_verified_record.json', record)
    print(f'Completed {label}: validation={record["mean"]:.6f}', flush=True)
    return record


def run_completion(args):
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError('output root must be a new directory')
    output.mkdir(parents=True)
    protocol = read_json(args.protocol)
    required = {'seeds': [42, 43, 44], 'width_order': [16, 14, 12], 'epochs': 20,
                'precision': 'float32', 'automatic_held_out_evaluation': False}
    if any(protocol.get(k) != v for k, v in required.items()):
        raise ValueError('registered protocol changed; refusing to run')
    write_json(output / 'registered_protocol.json', protocol)
    audit = audit_existing(args.existing_root, args.manifest, args.existing_accounting)
    write_json(output / 'existing_evidence_audit.json', audit)
    print('Existing A/B/D evidence verified; no repeated training', flush=True)
    profile = profile_resources(Path(args.manifest), Path(args.data_root), output / 'resource_probe', widths=(16, 14, 12), device='cuda')
    selection_profile = profile
    old = audit['reused_arms']
    # A microbenchmark cannot erase the recorded whole-training memory failure.
    if old['B']['vram'] > old['D']['vram'] * RULES['maximum_vram_ratio']:
        selection_profile = {**profile, 'rows': [r for r in profile['rows'] if not (r['model'] == 'improved' and r['base_channels'] == 16)]}
    width = select_resource_width(selection_profile, limit=RULES['maximum_vram_ratio'])
    baseline_aug = train_one(output, 'E_baseline_aug_seed42', 'baseline', 16, 42, args.data_root, args.manifest)
    old = audit['reused_arms']
    factorial = {**old, 'E': baseline_aug}
    contrasts = {'augmentation_on_baseline': baseline_aug['mean']-old['D']['mean'],
                 'augmentation_on_improved': old['B']['mean']-old['A']['mean'],
                 'architecture_without_augmentation': old['A']['mean']-old['D']['mean'],
                 'architecture_with_augmentation': old['B']['mean']-baseline_aug['mean']}
    contrasts['architecture_augmentation_interaction'] = contrasts['augmentation_on_improved']-contrasts['augmentation_on_baseline']
    write_json(output / 'factorial_validation_comparison.json', {'arms': factorial, 'contrasts': contrasts,
               'scope': 'validation only; architecture includes its auxiliary-loss objective'})
    decision = {'resource_selected_width': width, 'rules': RULES, 'automatic_test_permitted': False,
                'held_out_test_run': False, 'fixed_seed_policy': [42, 43, 44]}
    if width is None:
        decision['status'] = 'no_profiled_width_passed_memory_limit'
        write_json(output / 'completion_status.json', decision)
        return decision
    candidate = old['B'] if width == 16 else train_one(output, f'improved_width{width}_seed42', 'improved', width, 42, args.data_root, args.manifest)
    gates = {'against_augmented_baseline': validation_gate(candidate, baseline_aug),
             'against_original_D': validation_gate(candidate, old['D'])}
    decision['seed42_gates'] = gates
    decision['seed42_candidate_checkpoint_sha256'] = candidate['checkpoint_sha256']
    if not all(g['eligible_for_test_review'] for g in gates.values()):
        decision['status'] = 'seed42_candidate_failed_screen_no_extra_seeds'
        write_json(output / 'completion_status.json', decision)
        return decision
    rows = [{'seed': 42, 'baseline': baseline_aug, 'improved': candidate}]
    for seed in (43, 44):
        base = train_one(output, f'baseline_aug_seed{seed}', 'baseline', 16, seed, args.data_root, args.manifest)
        improved = train_one(output, f'improved_width{width}_seed{seed}', 'improved', width, seed, args.data_root, args.manifest)
        rows.append({'seed': seed, 'baseline': base, 'improved': improved})
        write_json(output / 'paired_seed_records.json', rows)
    summary = summarize_seed_pairs(rows)
    summary['per_seed_gates'] = {str(r['seed']): validation_gate(r['improved'], r['baseline']) for r in rows}
    summary['all_seed_gates_pass'] = all(g['eligible_for_test_review'] for g in summary['per_seed_gates'].values())
    write_json(output / 'paired_seed_summary.json', summary)
    decision['status'] = 'paired_validation_complete_await_frozen_test_plan'
    decision['stable_positive_validation_evidence'] = summary['all_seed_gates_pass'] and summary['all_paired_differences_positive']
    decision['test_readiness_review_only'] = decision['stable_positive_validation_evidence']
    write_json(output / 'completion_status.json', decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing-root', required=True, type=Path)
    parser.add_argument('--existing-accounting', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=Path('evidence/dataset_manifest.json'))
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=Path('configs/validation_completion_protocol.json'))
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_completion(args), indent=2), flush=True)


if __name__ == '__main__':
    main()
