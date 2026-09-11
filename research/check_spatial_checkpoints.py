"""Validate A/B/C short-run checkpoint routing and same-input inference."""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_fused_dataset import load_model

torch.set_num_threads(2)
base = Path('/root/autodl-fs/research_protocol/v1/runs/spatial_smoke_corrected_v1')
results = {}
for mode in ('none', 'independent', 'joint'):
    history = json.loads((base / mode / 'history.json').read_text())
    assert len(history) == 2
    model = load_model(base / mode / 'best.pth', torch.device('cuda'), False, False)
    aligner = model.sacafm_layers[0].aligner
    actual = type(aligner).__name__ if aligner is not None else 'None'
    assert actual == {'none': 'None', 'independent': 'IndependentDSDAM', 'joint': 'CrossModalDSDAM'}[mode]
    torch.manual_seed(100)
    x, y = torch.rand(1, 1, 128, 128, device='cuda'), torch.rand(1, 1, 128, 128, device='cuda')
    with torch.inference_mode():
        output = model(x, y)
    assert output.shape == x.shape and torch.isfinite(output).all()
    results[mode] = {'parameters': sum(p.numel() for p in model.parameters()),
                     'aligner_class': actual, 'epochs': len(history),
                     'updates': sum(h['optimizer_updates'] for h in history),
                     'skips': sum(h['skipped_updates'] for h in history),
                     'reloaded_forward': 'passed'}
    del model, aligner, x, y, output
    torch.cuda.empty_cache()
print(json.dumps(results, indent=2))
