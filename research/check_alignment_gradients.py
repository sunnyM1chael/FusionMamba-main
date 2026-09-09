"""Small CPU backward check of alignment only; not a full GPU training test."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from DSDAM import CrossModalDSDAM


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = CrossModalDSDAM(channels=16, num_heads=4, window_size=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ir, vis = torch.randn(2,16,16,16), torch.randn(2,16,16,16)
    target = torch.randn_like(ir)
    results = []
    for step in range(2):
        optimizer.zero_grad()
        a,b,confidence = model(ir,vis)
        assert a.shape == b.shape == ir.shape
        assert confidence.shape == (2,1,16,16)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert ((confidence >= 0)&(confidence <= 1)).all()
        loss = (a-target).square().mean()+(b-target).square().mean()+confidence.mean()*.01
        loss.backward()
        grads = {name:float(param.grad.abs().sum()) for name,param in model.named_parameters()
                 if param.grad is not None and ('offset_head' in name or 'joint_projection' in name)}
        assert all(torch.isfinite(param.grad).all() for param in model.parameters() if param.grad is not None)
        assert grads['offset_head.weight'] > 0
        # Zero-initialized offsets block projection gradients on the FIRST update.
        if step == 1:
            assert sum(v for k,v in grads.items() if 'joint_projection' in k) > 0
        optimizer.step()
        results.append({'step':step+1,'loss':float(loss.detach()),'gradient_l1':grads})
    print(json.dumps({'status':'passed','scope':'CPU CrossModalDSDAM only; full GPU test pending','steps':results},indent=2))


if __name__ == '__main__':
    main()
