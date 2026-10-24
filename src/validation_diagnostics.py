"""Render checkpoint-selected validation failure panels without opening test files."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import HipMRI3DDataset
from predict import load_checkpoint_model
from train import load_audited_pairs,prostate_diagnostics
from validation_visuals import render_validation_gif

def generate(checkpoint,manifest,data_root,output,device='cpu',count=5):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    device=torch.device(device);model,config=load_checkpoint_model(checkpoint,device)
    pairs,ids,n=load_audited_pairs(manifest)
    data=HipMRI3DDataset(pairs,data_root,ids['validation'],n,tuple(config['target_shape']))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    patient_scores={}; records=[]
    for index,record in enumerate(data.records):
        image,target=data[index]
        with torch.no_grad(): prediction=model(image[None].to(device)).argmax(1)[0].cpu()
        counts=prostate_diagnostics(prediction,target,config['prostate_label'])
        records.append({'index':index,'patient_id':record.patient_id,'case_id':record.case_id,**counts})
        patient_scores.setdefault(record.patient_id,[]).append(counts['prostate_dice'])
    means={p:sum(v)/len(v) for p,v in patient_scores.items()}
    worst=sorted(means,key=lambda p:(means[p],p))[:count]
    panels=[]
    for patient in worst:
        row=min((r for r in records if r['patient_id']==patient),key=lambda r:(r['prostate_dice'],r['case_id']))
        image,target=data[row['index']]
        with torch.no_grad(): prediction=model(image[None].to(device)).argmax(1)[0].cpu()
        actual=target.numpy()==config['prostate_label'];predicted=prediction.numpy()==config['prostate_label']
        error=actual^predicted;eligible=np.flatnonzero(actual.sum((1,2)))
        z=int(max(eligible,key=lambda z:error[z].sum())) if len(eligible) else int(error.sum((1,2)).argmax())
        fig,axes=plt.subplots(1,4,figsize=(12,3),constrained_layout=True)
        for ax,title in zip(axes,['MRI','GT prostate','Predicted prostate','FP red / FN cyan']):ax.imshow(image[0,z],cmap='gray');ax.set_title(title);ax.axis('off')
        axes[1].imshow(np.ma.masked_where(~actual[z],actual[z]),cmap='Greens',alpha=.6,vmin=0,vmax=1)
        axes[2].imshow(np.ma.masked_where(~predicted[z],predicted[z]),cmap='Greens',alpha=.6,vmin=0,vmax=1)
        rgba=np.zeros((*actual[z].shape,4));rgba[predicted[z]&~actual[z]]=(1,0,0,.8);rgba[actual[z]&~predicted[z]]=(0,1,1,.8);axes[3].imshow(rgba)
        fig.suptitle(f'Validation {row["case_id"]}: case Dice {row["prostate_dice"]:.4f}; worst-error GT slice {z}')
        path=output/f'{patient}_validation_failure.png';fig.savefig(path,dpi=150);plt.close(fig);panels.append(str(path))
    fixed=min(range(len(data)),key=lambda i:(data.records[i].patient_id,data.records[i].case_id))
    render_validation_gif(model,data,fixed,device,config['prostate_label'],output/'validation_best_epoch.gif',saved['epoch'])
    result={'scope':'validation only; not held-out failure evidence','checkpoint':str(checkpoint),'epoch':saved['epoch'],'per_patient_dice':means,'mean_patient_dice':sum(means.values())/len(means),'fn':sum(r['false_negative_voxels'] for r in records),'fp':sum(r['false_positive_voxels'] for r in records),'panel_selection':'five worst validation patients by mean case Dice; worst case per patient; maximal error slice inside GT extent','panels':panels,'cases':records}
    (output/'validation_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','manifest','data-root','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args();generate(a.checkpoint,a.manifest,a.data_root,a.output,a.device)
if __name__=='__main__':main()
