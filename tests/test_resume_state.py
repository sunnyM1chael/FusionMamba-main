"""A restart must continue optimizer, schedule, RNG and shuffled data order."""
from argparse import Namespace
import random

import numpy as np
import torch

from train import build_loader, load_checkpoint, save_checkpoint, set_seed


def test_checkpoint_continues_next_update(tmp_path):
    set_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    loader = build_loader(torch.arange(12), 4, 0, True, 42)

    def step():
        order = torch.cat(list(loader))
        x = torch.rand(4, 3) + random.random() + np.random.rand()
        optimizer.zero_grad()
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return order, {k: v.clone() for k, v in model.state_dict().items()}, scheduler.get_last_lr()

    step()
    path = tmp_path / 'last.pth'
    save_checkpoint(path, model, optimizer, scheduler, scaler, 0, 1.2, [], Namespace(), [loader])
    expected_order, expected_params, expected_lr = step()
    set_seed(999)
    epoch, best, history = load_checkpoint(path, model, optimizer, scheduler, scaler, 'cpu', [loader])
    actual_order, actual_params, actual_lr = step()
    assert (epoch, best, history) == (1, 1.2, [])
    torch.testing.assert_close(actual_order, expected_order, rtol=0, atol=0)
    for key in expected_params:
        torch.testing.assert_close(actual_params[key], expected_params[key], rtol=0, atol=0)
    assert actual_lr == expected_lr
