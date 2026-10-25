import json
from pathlib import Path
import pytest
import torch
import hipmri_resource_probe as probe


def test_width_selection_prefers_16_then_14_and_rejects_bad_baseline():
    rows = [{'model':'baseline','base_channels':16,'train_peak_vram_mb':100},
            *[{'model':'improved','base_channels':w,'train_peak_vram_mb':p} for w,p in [(12,90),(14,130),(16,136)]]]
    assert probe.select_resource_width({'rows':rows}) == 14
    rows[-1]['train_peak_vram_mb'] = 135
    assert probe.select_resource_width({'rows':rows}) == 16
    rows[0]['train_peak_vram_mb'] = float('nan')
    with pytest.raises(ValueError): probe.select_resource_width({'rows':rows})


def test_selection_refuses_missing_duplicates_and_nonfinite_candidates():
    baseline = {'model':'baseline','base_channels':16,'train_peak_vram_mb':100}
    for value in [0,-1,float('inf'),float('nan')]:
        with pytest.raises(ValueError):
            probe.select_resource_width({'rows':[{**baseline,'train_peak_vram_mb':value}]})
    with pytest.raises(ValueError): probe.select_resource_width({'rows':[baseline,baseline]})
    assert probe.select_resource_width({'rows':[baseline,{'model':'improved','base_channels':16,'train_peak_vram_mb':float('nan')}]}) is None


def test_load_input_selects_lexical_train_case_only_and_center_crop(tmp_path, monkeypatch):
    cases = [dict(case_id=c, patient_id=c, split=s, image_path=c+'.nii', mask_path=c+'mask.nii', shape=[2]*3,spacing=[1]*3,orientation=['R','A','S'],labels=list(range(6))) for c,s in [('test','test'),('z','train'),('a','train'),('val','validation')]]
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps({'summary':{'labels':list(range(6))},'prostate_label':5,'prostate_label_source':'original source','cases':cases}))
    seen=[]
    class Dataset:
        def __init__(self,pairs,root,ids,n,shape,**kw):
            seen.extend(p.case_id for p in pairs)
            assert ids == {'a'} and n == 6 and shape == (128,)*3
        def __getitem__(self,index):
            return torch.arange(128.).reshape(1,128,1,1).expand(1,128,128,128), torch.zeros(128,128,128,dtype=torch.long)
    monkeypatch.setattr(probe,'HipMRI3DDataset',Dataset)
    full, crop, target, meta = probe._load_profile_input(path,tmp_path)
    assert seen == ['a']
    assert full.shape == (1,1,128,128,128) and crop.shape == (1,1,96,96,96)
    assert target.shape == (1,96,96,96) and crop[0,0,0,0,0] == 16
    assert meta['case_id'] == 'a' and meta['prostate_label_source'] == 'original source'
    assert len(meta['manifest_sha256']) == 64


def test_cpu_profile_rejected_before_reading_manifest(tmp_path):
    with pytest.raises(ValueError,match='CUDA'):
        probe.profile_resources(tmp_path/'missing',tmp_path,tmp_path,device='cpu')


def test_source_fallback_verifies_frozen_local_evidence():
    source, digest = probe._verified_label_source()
    assert '10.25919/45T8-P065' in source and '5=Prostate' in source
    assert digest == '89c5ef8336573b6c359d0828e2ffcf6819149490fee20071f7436ed411dd89af'


def test_tiny_cpu_train_step_creates_adam_state_and_updates_weights():
    model = torch.nn.Conv3d(1,6,1)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    before = model.weight.detach().clone()
    probe._train_step(model,optimizer,torch.randn(1,1,2,2,2),torch.full((1,2,2,2),5,dtype=torch.long))
    assert not torch.equal(model.weight,before)
    assert optimizer.state and all(torch.isfinite(p).all() for p in model.parameters())


def test_probe_protocol_uses_initialized_adam_and_separate_inference(monkeypatch):
    events=[]
    class Tiny(torch.nn.Conv3d):
        def __init__(self, **kwargs): super().__init__(1,6,1)
        def forward(self,x):
            events.append('train' if self.training else 'infer')
            return super().forward(x)
    monkeypatch.setattr(probe,'BaselineUNet3D',Tiny)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device: events.append('sync'))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda device: events.append('reset'))
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda device: 1024**2)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda: None)
    full = torch.randn(1,1,2,2,2)
    row = probe._profile_model('baseline',16,full,full,torch.zeros(1,2,2,2,dtype=torch.long),torch.device('cpu'))
    assert events.count('train') == 5 and events.count('infer') == 8
    first_reset = events.index('reset')
    assert events[:first_reset].count('train') == 2
    assert events[:[i for i,x in enumerate(events) if x == 'reset'][1]].count('infer') == 3
    assert row['train_peak_vram_mb'] == row['inference_peak_vram_mb'] == 1
    assert len(row['inference_seconds']) == 5 and row['mean_inference_seconds'] > 0
