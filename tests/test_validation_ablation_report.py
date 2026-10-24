from validation_ablation_report import validation_gate

def test_validation_gate_rejects_fn_growth_despite_mean_gain():
    baseline=dict(mean=0.8,worst=0.7,fn=100,vram=100,time=1)
    candidate=dict(mean=0.82,worst=0.7,fn=120,vram=110,time=1)
    assert not validation_gate(candidate,baseline)['eligible_for_test_review']

def test_validation_gate_passes_declared_resource_and_accuracy_limits():
    baseline=dict(mean=0.8,worst=0.7,fn=100,vram=100,time=1)
    candidate=dict(mean=0.82,worst=0.7,fn=100,vram=110,time=1)
    assert validation_gate(candidate,baseline)['eligible_for_test_review']
