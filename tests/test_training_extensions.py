import torch
import pytest
from dataset import AugmentationConfig, augment_training_pair
from plot_training import render_learning_rate_curve

def test_spatial_alignment_and_integer_labels():
    mask = torch.arange(64).reshape(4,4,4).long()
    image = mask[None].float()
    config = AugmentationConfig(lr_flip=1, ap_flip=1, noise=0, blur=0, gamma=0)
    actual, labels = augment_training_pair(image, mask, config, ('R','A','S'))
    assert torch.equal(actual[0].long(), labels)
    assert torch.equal(labels, mask.flip((1,2)))
    assert labels.dtype == torch.long

def test_intensity_augmentation_never_changes_mask():
    image = torch.linspace(-2,2,64).reshape(1,4,4,4)
    mask = (image[0]>0).long()
    torch.manual_seed(4)
    actual, labels = augment_training_pair(image, mask, AugmentationConfig(lr_flip=0,ap_flip=0,noise=1,blur=1,gamma=1), ('R','A','S'))
    assert torch.equal(labels,mask)
    assert torch.isfinite(actual).all()
    assert not torch.equal(actual,image)

@pytest.mark.parametrize('value',[-0.1,1.1,float('nan')])
def test_reject_invalid_probability(value):
    with pytest.raises(ValueError): AugmentationConfig(noise=value)

def test_lr_plot_requires_real_lr(tmp_path):
    with pytest.raises(ValueError): render_learning_rate_curve({'old':[{'epoch':1}]},tmp_path/'lr.png')

def test_dataset_blocks_validation_augmentation(tmp_path):
    from dataset import HipMRI3DDataset
    with pytest.raises(ValueError, match='only for train'):
        HipMRI3DDataset([],tmp_path,set(),2,augmentation_config=AugmentationConfig(),split='validation')

def test_cosine_training_records_used_lr_and_validation_gifs(tmp_path, monkeypatch):
    import json
    from argparse import Namespace
    import numpy as np
    import nibabel as nib
    from train import run_training
    cases=[]
    for split, patient in [('train','A001'),('validation','B002'),('test','C003')]:
        image_path=f'semantic_MRs/{patient}.nii.gz'; mask_path=f'semantic_labels_only/{patient}.nii.gz'
        image=np.random.default_rng(2).normal(size=(16,16,16)).astype('float32')
        mask=np.zeros((16,16,16),dtype='int16'); mask[4:12,4:12,4:12]=1
        # Deliberately omit test files: successful training proves no test data was read.
        if split!='test':
            for path, array in [(image_path,image),(mask_path,mask)]:
                output=tmp_path/path; output.parent.mkdir(parents=True,exist_ok=True)
                nib.save(nib.Nifti1Image(array,np.eye(4)),output)
        cases.append(dict(case_id=patient,patient_id=patient,image_path=image_path,mask_path=mask_path,shape=[16]*3,spacing=[1]*3,orientation=['R','A','S'],labels=[0,1],split=split))
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps({'summary':{'labels':[0,1]},'cases':cases}))
    args=Namespace(manifest=manifest,data_root=tmp_path,output_dir=tmp_path/'run',model='improved',architecture_version='improved_context_v2',epochs=2,batch_size=1,learning_rate=0.001,base_channels=2,target_shape=(16,16,16),seed=42,device='cpu',prostate_label=1,prostate_label_source='synthetic fixture',augmentation=True,scheduler='cosine',eta_min=0.00001,validation_gifs=True,selection_metric='prostate_dice')
    import train
    original_train_epoch = train._train_epoch
    observed_epochs = []
    def observe_epoch(*positional, **keywords):
        if observed_epochs:
            assert (args.output_dir / 'training_curves.png').is_file()
            assert (args.output_dir / 'learning_rate_schedule.png').is_file()
        observed_epochs.append(len(observed_epochs) + 1)
        return original_train_epoch(*positional, **keywords)
    monkeypatch.setattr(train, '_train_epoch', observe_epoch)
    result=run_training(args)
    assert [r['learning_rate'] for r in result['history']]==pytest.approx([0.001,0.000505])
    output=args.output_dir
    config=json.loads((output/'experiment_config.json').read_text())
    assert config['augmentation_config']['lr_flip']==0.5
    assert config['validation_augmentation'] is False
    from PIL import Image
    with Image.open(output/'validation_best_epoch.gif') as gif: assert gif.n_frames>1
    checkpoint=torch.load(output/'best.pt',weights_only=False)
    assert checkpoint['scheduler_state_dict'] is not None
