"""One full FusionMamba CUDA forward/backward check; not a training result."""
import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.vmamba_Fusion_efficross import VSSM_Fusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weighting_mode', choices=('equal','learned','acgaw'), default='acgaw')
    parser.add_argument('--size', type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    model = VSSM_Fusion(use_dsdam=True, weighting_mode=args.weighting_mode).to(device).train()
    ir = torch.rand(1,1,args.size,args.size,device=device)
    vis = torch.rand_like(ir)
    torch.cuda.reset_peak_memory_stats()
    output = model(ir,vis)
    loss = (output-torch.maximum(ir,vis)).abs().mean()
    loss.backward()
    alignment = {n:float(p.grad.abs().sum()) for n,p in model.named_parameters()
                 if p.grad is not None and 'aligner.offset_head' in n}
    weighting = {n:float(p.grad.abs().sum()) for n,p in model.named_parameters()
                 if p.grad is not None and 'adaptive_weight' in n}
    assert output.shape == ir.shape and torch.isfinite(output).all()
    assert alignment and sum(alignment.values()) > 0
    if args.weighting_mode == 'equal':
        assert not weighting or sum(weighting.values()) == 0
    else:
        assert weighting and sum(weighting.values()) > 0
    print(json.dumps({'status':'passed','weighting_mode':args.weighting_mode,
                      'device':torch.cuda.get_device_name(0),
                      'shape':list(output.shape),'loss':float(loss.detach()),
                      'peak_memory_mib':round(torch.cuda.max_memory_allocated()/2**20,1),
                      'alignment_gradient_l1':sum(alignment.values()),
                      'weighting_gradient_l1':sum(weighting.values()) if weighting else 0.0},indent=2))


if __name__ == '__main__':
    main()
