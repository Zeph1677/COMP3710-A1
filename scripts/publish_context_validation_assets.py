"""Publish validation-only figures from retained job 692354; never load test images."""
from pathlib import Path
import argparse, hashlib, json, shutil, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from dataset import HipMRI3DDataset
from train import load_audited_pairs, select_validation_checkpoint, prostate_diagnostics
from predict import load_checkpoint_model
from compare_failures import error_relevant_prostate_slice
from plot_training import render_training_curves, render_learning_rate_curve


def digest(path):
    with Path(path).open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def publish_curves(histories, out):
    """Keep selected epochs in legends so annotations cannot overlap."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for arm, history in histories.items():
        row = select_validation_checkpoint(history, 'prostate_dice')
        epochs = [r['epoch'] for r in history]
        line = axes[0].plot(epochs, [r['train_loss'] for r in history], marker='o', label=arm)[0]
        axes[1].plot(epochs, [r['validation_patient_prostate_dice'] for r in history], marker='o', color=line.get_color(), label=f"{arm} (selected e{int(row['epoch'])})")
        axes[1].scatter([row['epoch']], [row['validation_patient_prostate_dice']], s=75, color=line.get_color(), zorder=4)
    axes[0].set(title='Training loss', xlabel='Epoch', ylabel='Composite loss')
    axes[1].set(title='Validation patient prostate Dice', xlabel='Epoch', ylabel='Dice')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    figure.savefig(out/'training_loss_and_validation_dice.png', dpi=180);plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=Path('artifacts/2026-10-25/692354_context_v2_ablation/context_v2_ablation_692354/runs'))
    parser.add_argument('--output',type=Path,default=Path('docs/assets/context_v2_ablation_692354'))
    parser.add_argument('--data-root',type=Path,default=Path('data/HipMRI_Study_open'))
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();root=args.run_root;out=args.output;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    configs={};histories={};selected={};provenance={'scope':'validation only','job':'692354','raw_files':{},'assets':{},'device':args.device}
    frozen_fields=['manifest_sha256','seed','target_shape','train_patch_shape','epochs','batch_size','base_channels','prostate_label','prostate_ce_weight','prostate_dice_weight','selection_metric']
    for arm in 'ABCD':
        directory=root/arm;config=json.loads((directory/'experiment_config.json').read_text());history=json.loads((directory/'history.json').read_text());summary=json.loads((directory/'training_summary.json').read_text())
        assert [r['epoch'] for r in history]==list(range(1,21)),f'incomplete {arm}'
        row=select_validation_checkpoint(history,'prostate_dice');assert row['epoch']==summary['selected_epoch']
        checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=False);assert checkpoint['epoch']==row['epoch'];assert checkpoint['config']==config
        configs[arm]=config;histories[arm]=history;selected[arm]=row
        for name in ['best.pt','experiment_config.json','history.json','training_summary.json','validation_visualization.json']:
            provenance['raw_files'][f'{arm}/{name}']=digest(directory/name)
        for name in ['validation_epoch_1.gif','validation_epoch_5.gif','validation_best_epoch.gif']:
            source=directory/name;destination=out/f'{arm}_{name}';shutil.copyfile(source,destination)
            with Image.open(destination) as gif:
                assert gif.n_frames>1;provenance['assets'][destination.name]={'source':str(source),'sha256':digest(source),'frames':gif.n_frames,'size':list(gif.size)}
        print(f'Audited {arm}: selected epoch {row["epoch"]}',flush=True)
    for arm in 'ABC':assert all(configs[arm][f]==configs['D'][f] for f in frozen_fields)
    manifest=Path('evidence/dataset_manifest.json');assert digest(manifest)==configs['B']['manifest_sha256']
    assert digest(root.parent/'re_audited_manifest.json')==digest(manifest)
    publish_curves(histories,out)
    render_learning_rate_curve(histories,out/'learning_rate_schedule.png')
    patients=sorted(selected['D']['validation_patient_prostate_dice_by_patient'])
    figure,ax=plt.subplots(figsize=(10,4),constrained_layout=True);x=np.arange(len(patients))
    for i,arm in enumerate('ABCD'):ax.bar(x+(i-1.5)*.2,[selected[arm]['validation_patient_prostate_dice_by_patient'][p] for p in patients],width=.2,label=arm)
    ax.set(xticks=x,xticklabels=patients,ylabel='Patient mean prostate Dice',ylim=(0.6,1),title='Validation: patients at each arm\'s selected checkpoint (axis starts at 0.6)');ax.legend(ncol=4);ax.grid(axis='y',alpha=.2);figure.savefig(out/'patient_validation_dice.png',dpi=160);plt.close(figure)
    # Preserve original slice-sweep GIFs; compose synchronized same-case comparisons.
    for name,arms,suffixes in [('best_ABCD',['A','B','C','D'],['validation_best_epoch.gif']*4),('B_progress',['B']*3,['validation_epoch_1.gif','validation_epoch_5.gif','validation_best_epoch.gif'])]:
        sources=[Image.open(out/f'{a}_{suffix}') for a,suffix in zip(arms,suffixes)]
        assert len({g.n_frames for g in sources})==1 and len({g.size for g in sources})==1
        frames=[];width,height=sources[0].size
        for index in range(sources[0].n_frames):
            frame=Image.new('RGB',(width,(height+24)*len(sources)),'white');draw=ImageDraw.Draw(frame)
            for k,(arm,gif,suffix) in enumerate(zip(arms,sources,suffixes)):
                gif.seek(index);epoch=1 if 'epoch_1.' in suffix else 5 if 'epoch_5.' in suffix else int(selected[arm]['epoch'])
                draw.text((12,k*(height+24)+4),f'Arm {arm} | epoch {epoch} | validation G021_Week0 | spatial slice sweep',fill='black')
                frame.paste(gif.convert('RGB'),(0,k*(height+24)+24))
            frames.append(frame)
        frames[0].save(out/f'{name}.gif',save_all=True,append_images=frames[1:],duration=140,loop=0)
        frames[len(frames)//2].save(out/f'{name}_preview.png')
        for source in sources:source.close()
    # Recompute B/D on validation only; do not modify original recorded selection metrics.
    pairs,ids,n=load_audited_pairs(manifest);data=HipMRI3DDataset(pairs,args.data_root,ids['validation'],n,tuple(configs['B']['target_shape']))
    models={arm:load_checkpoint_model(root/arm/'best.pt',torch.device(args.device))[0] for arm in ['B','D']}
    records=[]
    for index,record in enumerate(data.records):
        image,target=data[index];predictions={}
        for arm,model in models.items():
            with torch.no_grad():predictions[arm]=model(image[None].to(args.device)).argmax(1)[0].cpu()
        counts={arm:prostate_diagnostics(pred,target,configs['B']['prostate_label']) for arm,pred in predictions.items()}
        records.append({'index':index,'patient_id':record.patient_id,'case_id':record.case_id,'B':counts['B'],'D':counts['D'],'common_error':sum(v['false_positive_voxels']+v['false_negative_voxels'] for v in counts.values())/2})
        print(f'Validation inference {index+1}/{len(data)} {record.case_id}',flush=True)
    chosen=[];seen=set()
    for row in sorted(records,key=lambda r:(-r['common_error'],r['patient_id'],r['case_id'])):
        if row['patient_id'] not in seen:chosen.append(row);seen.add(row['patient_id'])
        if len(chosen)==5:break
    for row in chosen:
        image,target=data[row['index']];predictions={}
        for arm,model in models.items():
            with torch.no_grad():predictions[arm]=model(image[None].to(args.device)).argmax(1)[0].cpu()
        z=error_relevant_prostate_slice(target,predictions['D'],predictions['B'],configs['B']['prostate_label']);actual=target[z].numpy()==configs['B']['prostate_label']
        figure,axes=plt.subplots(1,6,figsize=(16,3.4),constrained_layout=True)
        for ax,title in zip(axes,['MRI','GT prostate','D Baseline','B augmented Context v2','D: FP red / FN cyan','B: FP red / FN cyan']):ax.imshow(image[0,z],cmap='gray');ax.set_title(title,fontsize=9);ax.axis('off')
        axes[1].imshow(np.ma.masked_where(~actual,actual),cmap='Greens',alpha=.6,vmin=0,vmax=1)
        for arm,column in [('D',2),('B',3)]:
            predicted=predictions[arm][z].numpy()==configs['B']['prostate_label'];axes[column].imshow(np.ma.masked_where(~predicted,predicted),cmap='Greens',alpha=.6,vmin=0,vmax=1)
            rgba=np.zeros((*actual.shape,4));rgba[predicted&~actual]=(1,0,0,.8);rgba[actual&~predicted]=(0,1,1,.8);axes[column+2].imshow(rgba)
        figure.suptitle(f"VALIDATION {row['case_id']} | slice {z} | whole-case Dice D={row['D']['prostate_dice']:.4f}, B={row['B']['prostate_dice']:.4f}",fontsize=11)
        row['slice_index']=z;row['panel']=f"failure_{row['case_id']}.png";figure.savefig(out/row['panel'],dpi=150);plt.close(figure)
    result={'scope':'validation only; no held-out inference','selection_rule':'descending mean B+D whole-case FP+FN; one case per distinct patient; then patient ID/case ID; highest combined error slice inside GT prostate extent','cases':records,'selected_cases':chosen}
    (out/'paired_validation_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    provenance['raw_git_sha']=(root.parent/'git_sha.txt').read_text().strip()
    provenance['scheduler_accounting']='No final sacct file found in synchronized source package; not independently verified'
    for file in out.iterdir():
        if file.suffix in {'.png','.gif'}:provenance['assets'].setdefault(file.name,{'sha256':digest(file)})
    (out/'asset_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    cards=''.join(f'<section><h2>{a}: epoch 1 / epoch 5 / selected epoch {int(selected[a]["epoch"])}</h2>'+''.join(f'<a href="{a}_{s}"><img loading="lazy" src="{a}_{s}" alt="Validation {a} {s}"></a>' for s in ['validation_epoch_1.gif','validation_epoch_5.gif','validation_best_epoch.gif'])+'</section>' for a in 'ABCD')
    html='<!doctype html><meta charset="utf-8"><title>Context v2 validation evidence</title><style>body{max-width:1200px;margin:30px auto;font:16px system-ui;color:#172b4d}img{max-width:100%;height:auto}section{border-top:1px solid #ccd;padding:16px 0}p{line-height:1.7}</style><h1>Job 692354: validation evidence</h1><p>All GIFs show validation G021_Week0. Frames sweep spatial slices, not training epochs. Each original GIF is copied byte-for-byte from the retained Rangpur output. B is best on mean validation Dice; no candidate passes the memory gate. No held-out test was run.</p><h2>B: training checkpoints</h2><img src="B_progress.gif"><h2>A/B/C/D: selected checkpoints</h2><img src="best_ABCD.gif">'+cards+'<h2>Matched D/B validation failure review</h2>'+''.join(f'<img loading="lazy" src="{r["panel"]}">' for r in chosen)
    (out/'gallery.html').write_text(html,encoding='utf-8')
    print('Completed validation presentation package',flush=True)
if __name__=='__main__':main()
