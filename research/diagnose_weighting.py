"""Frozen-validation inference diagnostics; never trains or reads test images."""
import json
import time
import hashlib
from pathlib import Path
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from TaskFusion_dataset import Fusion_dataset
from models.vmamba_Fusion_efficross import VSSM_Fusion
from models.cross import AlignmentConfidenceGuidedAdaptiveWeighting
from loss import Fusionloss

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
manifest = Path('/root/autodl-fs/research_protocol/v1/development_splits/msrs/val.txt')
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == 'c234c286a0ea4065011f6bd572541fe400f5bba970a3f91169704448dbe3e940'
dataset = Fusion_dataset('val', ir_path='/root/autodl-fs/datasets/MSRS/train/ir',
                         vi_path='/root/autodl-fs/datasets/MSRS/train/vi', split_file=str(manifest))
assert len(dataset) == 109
assert torch.cuda.is_available()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
criterion = Fusionloss().cuda().eval()
runs = {'equal': 'msrs_sacafm_equal_seed42_controls_v1',
        'learned': 'msrs_sacafm_learned_seed42_controls_v1',
        'acgaw': 'msrs_sacafm_acgaw_seed42_e2cbf9f'}
start = time.monotonic()
for mode, run in runs.items():
    checkpoint = torch.load(Path('/root/autodl-fs/research_protocol/v1/runs') / run / 'best.pth', map_location='cpu', weights_only=False)
    model = VSSM_Fusion(use_dsdam=True, weighting_mode=mode).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    gates = {n: m for n, m in model.named_modules() if isinstance(m, AlignmentConfidenceGuidedAdaptiveWeighting)}
    current = {}
    handles = []
    def hook(name):
        def record(module, inputs, output):
            w = output[1][:, 0].detach()
            confidence = inputs[2] if len(inputs) > 2 else None
            current[name] = {'mean_ir': w.mean().item(), 'std_ir': w.std().item(),
                             'abs_from_half': (w - .5).abs().mean().item(),
                             'fraction_near_half': ((w - .5).abs() < .01).float().mean().item(),
                             'confidence_mean': confidence.mean().item() if confidence is not None else None}
        return record
    for name, gate in gates.items():
        handles.append(gate.register_forward_hook(hook(name)))
    rows = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            vis, ir = dataset[index]
            vis, ir = vis[None].cuda(), ir[None].cuda()
            height, width = ir.shape[-2:]
            pad = (0, (-width) % 32, 0, (-height) % 32)
            a, b = F.pad(ir, pad, mode='replicate'), F.pad(vis, pad, mode='replicate')
            pred = model(a, b)[..., :height, :width]
            assert torch.isfinite(pred).all()
            losses = criterion(vis, ir, None, pred, 0)
            row = {'name': dataset.filenames_ir[index], 'loss': [x.item() for x in losses], 'gates': dict(current)}
            if index < 8:
                if mode != 'equal':
                    for gate in gates.values():
                        gate.mode = 'equal'
                    alternative = model(a, b)[..., :height, :width]
                    row['inference_only_equal_gate_MAE'] = (pred - alternative).abs().mean().item()
                    for gate in gates.values():
                        gate.mode = mode
                panel = torch.cat((ir, vis, pred), dim=-1)[0, 0].cpu().numpy()
                assert cv2.imwrite(str(out / f'{mode}_{index:02d}.png'), (panel * 255).round().clip(0, 255).astype(np.uint8))
            rows.append(row)
            if (index + 1) % 25 == 0:
                print(mode, index + 1, 'elapsed_seconds', round(time.monotonic() - start, 1), flush=True)
    params = {n: {'temperature': m.temperature.item(), 'saliency_gain': m.saliency_gain.item()} for n, m in gates.items()}
    payload = {'mode': mode, 'rows': rows, 'parameters': params, 'elapsed_seconds': time.monotonic() - start,
               'mean_loss': np.mean([r['loss'] for r in rows], axis=0).tolist(),
               'note': 'First eight manifest images fixed in advance. Equal-gate intervention is inference sensitivity, not retrained ablation.'}
    (out / f'{mode}.json').write_text(json.dumps(payload, indent=2))
    print(mode, 'COMPLETE', payload['mean_loss'], flush=True)
    for handle in handles:
        handle.remove()
    del model, gates
    torch.cuda.empty_cache()
print('ALL_COMPLETE', round(time.monotonic() - start, 1), flush=True)
