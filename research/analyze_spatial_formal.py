"""Final clean and pre-declared synthetic-shift analysis on validation only."""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from TaskFusion_dataset import Fusion_dataset
from loss import Fusionloss
from models.vmamba_Fusion_efficross import VSSM_Fusion

ROOT = Path('/root/autodl-fs/research_protocol/v1/runs/spatial_formal_corrected_seed42_v1')
OUT = ROOT / 'formal_validation_analysis_v1.json'
if OUT.exists():
    raise FileExistsError(OUT)
manifest = Path('/root/autodl-fs/research_protocol/v1/development_splits/msrs/val.txt')
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == 'c234c286a0ea4065011f6bd572541fe400f5bba970a3f91169704448dbe3e940'
dataset = Fusion_dataset('val', ir_path='/root/autodl-fs/datasets/MSRS/train/ir',
                         vi_path='/root/autodl-fs/datasets/MSRS/train/vi', split_file=str(manifest))
assert len(dataset) == 109 and torch.cuda.is_available()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
criterion = Fusionloss().cuda().eval()
modes = ('none', 'independent', 'joint')
conditions = [('clean', 0)] + [(axis, amount) for amount in (2, 4, 8) for axis in ('right', 'down')]


def shifted(x, axis, amount):
    if amount == 0:
        return x
    if axis == 'right':
        return F.pad(x[..., :-amount], (amount, 0, 0, 0), mode='replicate')
    return F.pad(x[..., :-amount, :], (0, 0, amount, 0), mode='replicate')


def boot_ci(values, seed=20260911, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        means[start:start+count] = values[rng.integers(0, len(values), (count, len(values)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


result = {'protocol': {'dataset': 'MSRS frozen validation only', 'pairs': 109,
                       'conditions': conditions, 'shifted_modality': 'visible',
                       'shift_padding': 'replicate', 'shift_evaluation':
                       'loss against original aligned IR/VIS, cropped by 8 pixels on every boundary',
                       'warning': 'Synthetic sensitivity/recovery test, not real unregistered data or geometric ground truth.',
                       'bootstrap': 'paired image resampling, 10000 draws, seed 20260911'}, 'modes': {}}
start_time = time.monotonic()
for mode in modes:
    folder = ROOT / mode
    history = json.loads((folder / 'history.json').read_text())
    assert len(history) == 100 and all(x['optimizer_updates'] == 243 and x['skipped_updates'] == 0 for x in history)
    checks = {}
    for filename in ('best.pth', 'last.pth'):
        checkpoint = torch.load(folder / filename, map_location='cpu', weights_only=False)
        state = checkpoint['model']
        checks[filename] = {'epoch_zero_based': checkpoint['epoch'], 'tensors': len(state),
                            'finite': all(bool(torch.isfinite(v).all()) for v in state.values() if torch.is_tensor(v)),
                            'resumable': checkpoint.get('optimizer') is not None}
        assert checks[filename]['finite']
        if filename == 'best.pth':
            best_state = state
        del checkpoint
    model = VSSM_Fusion(use_dsdam=True, spatial_mode=mode, weighting_mode='equal').cuda().eval()
    model.load_state_dict(best_state, strict=True)
    del best_state
    rows = []
    by_condition = {f'{a}_{n}': [] for a, n in conditions}
    with torch.inference_mode():
        for index in range(len(dataset)):
            vis, ir = dataset[index]
            vis, ir = vis[None].cuda(), ir[None].cuda()
            height, width = ir.shape[-2:]
            pad = (0, (-width) % 32, 0, (-height) % 32)
            row = {'name': dataset.filenames_ir[index], 'losses': {}}
            for axis, amount in conditions:
                input_vis = shifted(vis, axis, amount)
                output = model(F.pad(ir, pad, mode='replicate'),
                               F.pad(input_vis, pad, mode='replicate'))[..., :height, :width]
                assert torch.isfinite(output).all()
                if amount:
                    target_vis, target_ir, output_eval = vis[..., 8:-8, 8:-8], ir[..., 8:-8, 8:-8], output[..., 8:-8, 8:-8]
                else:
                    target_vis, target_ir, output_eval = vis, ir, output
                losses = [float(x) for x in criterion(target_vis, target_ir, None, output_eval, 0)]
                key = f'{axis}_{amount}'
                row['losses'][key] = losses
                by_condition[key].append(losses)
            rows.append(row)
            if (index + 1) % 25 == 0:
                print(mode, index + 1, round(time.monotonic() - start_time, 1), flush=True)
    best = min(history, key=lambda x: x['val_total'])
    summary = {}
    for key, values in by_condition.items():
        array = np.asarray(values)
        summary[key] = {'mean': array.mean(0).tolist(), 'total_bootstrap_ci95': boot_ci(array[:, 0])}
    result['modes'][mode] = {'best_epoch': best['epoch'], 'best_val': best['val_total'],
                             'last10_mean': statistics.mean(x['val_total'] for x in history[-10:]),
                             'last10_std': statistics.pstdev(x['val_total'] for x in history[-10:]),
                             'checkpoint_checks': checks, 'condition_summary': summary, 'rows': rows}
    print('COMPLETE', mode, flush=True)
    del model
    torch.cuda.empty_cache()

for challenger in ('independent', 'joint'):
    paired = {}
    for axis, amount in conditions:
        key = f'{axis}_{amount}'
        delta = [result['modes'][challenger]['rows'][i]['losses'][key][0] -
                 result['modes']['none']['rows'][i]['losses'][key][0] for i in range(109)]
        paired[key] = {'mean_delta_challenger_minus_none': statistics.mean(delta),
                       'ci95': boot_ci(delta), 'challenger_wins': sum(x < 0 for x in delta)}
    result.setdefault('paired_against_none', {})[challenger] = paired
delta = [result['modes']['joint']['rows'][i]['losses']['clean_0'][0] -
         result['modes']['independent']['rows'][i]['losses']['clean_0'][0] for i in range(109)]
result['paired_joint_minus_independent_clean'] = {'mean_delta': statistics.mean(delta),
                                                  'ci95': boot_ci(delta),
                                                  'joint_wins': sum(x < 0 for x in delta)}
result['elapsed_seconds'] = time.monotonic() - start_time
OUT.write_text(json.dumps(result, indent=2))
print('ALL_ANALYSIS_COMPLETE', round(result['elapsed_seconds'], 1), OUT, flush=True)
