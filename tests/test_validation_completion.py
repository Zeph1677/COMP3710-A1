from pathlib import Path
import json
import pytest
from validation_completion import make_training_arguments, summarize_seed_pairs, validate_final_accounting, audit_existing

def test_training_seed_does_not_change_manifest_or_protocol(tmp_path):
    args=make_training_arguments('baseline',True,16,43,tmp_path/'run',tmp_path/'data',Path('evidence/dataset_manifest.json'))
    assert args.seed==43 and args.manifest.as_posix()=='evidence/dataset_manifest.json'
    assert args.augmentation and args.epochs==20 and args.learning_rate==.001
    assert args.target_shape==[128]*3 and args.train_patch_shape==[96]*3
    assert args.selection_metric=='prostate_dice' and args.scheduler=='none'

@pytest.mark.parametrize('state', ['RUNNING','FAILED','CANCELLED'])
def test_accounting_rejects_nonfinal_success(state):
    with pytest.raises(ValueError): validate_final_accounting(f'692354|job|{state}|0:0|01:42:37|gpu=1')

def test_accounting_accepts_exact_job_success_only():
    assert validate_final_accounting('692354|job|COMPLETED|0:0|01:42:37|gpu=1')['elapsed']=='01:42:37'
    with pytest.raises(ValueError):validate_final_accounting('692354|job|COMPLETED|1:0|01:42:37|gpu=1')
    with pytest.raises(ValueError):validate_final_accounting('123456|job|COMPLETED|0:0|01:42:37|gpu=1')

def test_seed_summary_uses_paired_differences_and_sample_sd():
    rows=[{'seed':s,'baseline':{'mean':b},'improved':{'mean':i}} for s,b,i in [(42,.80,.82),(43,.81,.82),(44,.79,.81)]]
    summary=summarize_seed_pairs(rows)
    assert summary['paired_difference_mean']==pytest.approx(.05/3)
    assert summary['paired_difference_sample_sd']==pytest.approx(.00577350269)
    assert summary['seeds']==[42,43,44]
    with pytest.raises(ValueError):summarize_seed_pairs(rows[:2])

def test_audit_refuses_incomplete_evidence(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)):
        audit_existing(tmp_path/'runs',Path('evidence/dataset_manifest.json'),tmp_path/'sacct.txt')
