"""Fixed validation-only full-volume slice animations."""
from pathlib import Path
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def render_validation_gif(model, dataset, index, device, prostate_label, output_path, epoch):
    image, target = dataset[index]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(image[None].to(device))
        if isinstance(logits, tuple): logits = logits[0]
        predicted = logits.argmax(1)[0].cpu().numpy() == prostate_label
    model.train(was_training)
    actual = target.numpy() == prostate_label
    volume = image[0].numpy()
    low, high = np.percentile(volume, [1,99])
    frames = []
    for z in range(0,volume.shape[0],max(1,volume.shape[0]//24)):
        figure, axes = plt.subplots(1,4,figsize=(12,3),constrained_layout=True)
        for axis, title in zip(axes,['MRI','Ground truth prostate','Predicted prostate','FP red / FN cyan']):
            axis.imshow(volume[z],cmap='gray',vmin=low,vmax=high); axis.set_title(title); axis.axis('off')
        axes[1].imshow(np.ma.masked_where(~actual[z],actual[z]),cmap='Greens',vmin=0,vmax=1,alpha=0.6)
        axes[2].imshow(np.ma.masked_where(~predicted[z],predicted[z]),cmap='Greens',vmin=0,vmax=1,alpha=0.6)
        overlay = np.zeros((*actual[z].shape,4))
        overlay[predicted[z]&~actual[z]] = (1,0,0,0.8)
        overlay[actual[z]&~predicted[z]] = (0,1,1,0.8)
        axes[3].imshow(overlay)
        figure.suptitle(f'Validation {dataset.records[index].case_id} | epoch {epoch} | slice {z}')
        buffer=io.BytesIO(); figure.savefig(buffer,format='png',dpi=90); plt.close(figure)
        frames.append(Image.open(buffer).convert('RGB'))
    output=Path(output_path); output.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=140,loop=0)
