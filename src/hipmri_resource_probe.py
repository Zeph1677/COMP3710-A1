"""Matched CUDA allocated-memory probe; never evaluates held-out MRI volumes."""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch

from dataset import HipMRI3DDataset
from modules import BaselineUNet3D, ImprovedUNet3D
from train import composite_3d_loss, load_audited_pairs


def select_resource_width(profile: dict, limit: float = 1.35) -> int | None:
    """Return first eligible context width, comparing training allocated peaks."""
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('limit must be positive and finite')
    rows = profile['rows']
    baselines = [r for r in rows if r['model'] == 'baseline' and r['base_channels'] == 16]
    if len(baselines) != 1:
        raise ValueError('exactly one baseline16 is required')
    baseline = baselines[0]['train_peak_vram_mb']
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError('baseline peak must be positive and finite')
    for width in (16, 14, 12):
        candidates = [r for r in rows if r['model'] == 'improved' and r['base_channels'] == width]
        if len(candidates) > 1:
            raise ValueError('duplicate candidate width')
        if candidates:
            peak = candidates[0]['train_peak_vram_mb']
            if math.isfinite(peak) and peak > 0 and peak / baseline <= limit:
                return width
    return None


def _verified_label_source():
    path = Path(__file__).resolve().parents[1] / 'evidence/sources/csiro-brief-data-description.txt'
    text = path.read_text(encoding='utf-8')
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if digest != '89c5ef8336573b6c359d0828e2ffcf6819149490fee20071f7436ed411dd89af' or '5=Prostate' not in text:
        raise ValueError('frozen CSIRO label source verification failed')
    return 'CSIRO, Labelled weekly MR images of the male pelvis, v2, DOI: 10.25919/45T8-P065, BriefDataDescription.txt, Labels block: 5=Prostate.', digest


def _load_profile_input(manifest: Path, data_root: Path):
    raw = Path(manifest).read_bytes()
    config = json.loads(raw)
    pairs, splits, classes = load_audited_pairs(manifest)
    if classes != 6:
        raise ValueError('resource protocol requires six classes')
    verified_source, source_digest = _verified_label_source()
    source = next((section.get('prostate_label_source') for section in (config, config.get('config', {}), config.get('summary', {})) if section.get('prostate_label_source')), verified_source)
    label = next((section.get('prostate_label') for section in (config, config.get('config', {}), config.get('summary', {})) if 'prostate_label' in section), 5)
    if label != 5 or not isinstance(source, str) or not source.strip():
        raise ValueError('manifest must declare original prostate label source and prostate=5')
    candidates = sorted((p for p in pairs if p.patient_id in splits['train']), key=lambda p: p.case_id)
    if not candidates:
        raise ValueError('manifest has no train cases')
    pair = candidates[0]
    for path in (pair.image_path, pair.mask_path):
        if path.is_absolute() or not (Path(data_root).resolve() / path).resolve().is_relative_to(Path(data_root).resolve()):
            raise ValueError('case paths must stay within data_root')
    image, mask = HipMRI3DDataset([pair], data_root, {pair.patient_id}, classes, (128,)*3, split='train')[0]
    crop = image[:,16:112,16:112,16:112].contiguous()[None]
    target = mask[16:112,16:112,16:112].contiguous()[None]
    return image[None], crop, target, {'case_id':pair.case_id,'patient_id':pair.patient_id,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'num_classes':6,'prostate_label':5,'prostate_label_source':source,'label_source_normalized_sha256':source_digest,'train_input_shape':list(crop.shape),'inference_input_shape':list(image[None].shape)}


def _train_step(model, optimizer, image, target):
    optimizer.zero_grad(set_to_none=True)
    loss = composite_3d_loss(model(image), target, 6, 5, 4.0, 4.0)
    loss.backward()
    optimizer.step()


def _profile_model(name, width, full, crop, target, device):
    model = optimizer = image = labels = volume = None
    try:
        cls = BaselineUNet3D if name == 'baseline' else ImprovedUNet3D
        model = cls(in_channels=1,num_classes=6,base_channels=width).to(device=device,dtype=torch.float32)
        count = sum(p.numel() for p in model.parameters())
        image, labels = crop.to(device=device,dtype=torch.float32), target.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        model.train()
        for _ in range(2):
            _train_step(model,optimizer,image,labels)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(3):
            _train_step(model,optimizer,image,labels)
        torch.cuda.synchronize(device)
        train_peak = torch.cuda.max_memory_allocated(device) / 1024**2
        optimizer.zero_grad(set_to_none=True)
        optimizer = image = labels = None
        gc.collect()
        torch.cuda.empty_cache()
        model.eval()
        volume = full.to(device=device,dtype=torch.float32)
        with torch.inference_mode():
            for _ in range(3):
                model(volume)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            timings = []
            for _ in range(5):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                model(volume)
                torch.cuda.synchronize(device)
                timings.append(time.perf_counter()-started)
        infer_peak = torch.cuda.max_memory_allocated(device) / 1024**2
        return {'model':name,'architecture_version':getattr(model,'architecture_version','baseline_unet3d'),'base_channels':width,'parameter_count':count,'train_peak_vram_mb':train_peak,'inference_peak_vram_mb':infer_peak,'mean_inference_seconds':statistics.mean(timings),'std_inference_seconds':statistics.pstdev(timings),'inference_seconds':timings}
    finally:
        model = optimizer = image = labels = volume = None
        gc.collect()
        torch.cuda.empty_cache()


def profile_resources(manifest: Path, data_root: Path, output: Path, widths: tuple = (16,14,12), device: str = 'cuda') -> dict:
    """Profile actual CUDA peaks and synchronized volume latency; write JSON evidence.

    MB fields use MiB (1024**2 bytes). Training includes initialized Adam states;
    inference excludes optimizer/gradient tensors and excludes I/O/preprocessing.
    Global backend settings are observed, never changed. RNG states are restored.
    """
    selected = torch.device(device)
    if selected.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('real resource profiling requires CUDA')
    if not widths or len(set(widths)) != len(widths) or any(type(w) is not int or w not in (16,14,12) for w in widths):
        raise ValueError('widths must be unique members of (16,14,12)')
    full,crop,target,metadata = _load_profile_input(Path(manifest),Path(data_root))
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    metadata.update({'schema_version':1,'dtype':'float32','amp':False,'batch_size':1,'optimizer':'Adam','learning_rate':0.001,'prostate_ce_weight':4.0,'prostate_dice_weight':4.0,'train_warmup_steps':2,'train_measured_steps':3,'inference_warmup_steps':3,'inference_measured_steps':5,'memory_unit':'MiB','memory_measurement':'torch.cuda.max_memory_allocated','latency_std':'population','torch_version':str(torch.__version__),'cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),'cudnn_enabled':torch.backends.cudnn.enabled,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'gpu_name':props.name,'gpu_total_memory_bytes':props.total_memory,'gpu_compute_capability':[props.major,props.minor],'gpu_multiprocessor_count':props.multi_processor_count,'device':str(selected),'platform':platform.platform(),'processor':platform.processor()})
    rows = []
    with torch.random.fork_rng(devices=[index]), torch.autocast(device_type='cuda', enabled=False):
        for name,width in [('baseline',16),*[('improved',w) for w in widths]]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.random.default_generator.manual_seed(42)
            torch.cuda.default_generators[index].manual_seed(42)
            rows.append(_profile_model(name,width,full,crop,target,selected))
    metadata['rows'] = rows
    folder = Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'resource_profile.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return metadata
