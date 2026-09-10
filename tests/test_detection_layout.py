"""Check spatial/channel correspondence through the actual DSDAM forward."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


class IgnoreOffsets(nn.Module):
    def forward(self, x, offsets):
        return x


class ValuesOnly(nn.Module):
    def forward(self, x):
        return torch.cat((torch.zeros_like(x), torch.zeros_like(x), x), dim=1)


@pytest.mark.parametrize('window', [0, 1, 2])
def test_attention_preserves_channel_pixel_order(window):
    pytest.importorskip('torchvision')
    path = Path(__file__).resolve().parents[1] / 'yolo11_dsdam_deploy/ultralytics/nn/modules/DSDAM.py'
    spec = importlib.util.spec_from_file_location('layout_dsdam', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.DSDAM(4, 4, num_heads=2, window_size=window, energy_guided=False)
    model.conv_init = model.out_proj = model.norm = nn.Identity()
    model.deform_conv = IgnoreOffsets()
    model.qkv_proj = ValuesOnly()
    x = torch.arange(1, 65, dtype=torch.float32).reshape(1, 4, 4, 4)
    expected = torch.empty_like(x)
    size = window or 4
    for row in range(0, 4, size):
        for col in range(0, 4, size):
            block = x[:, :, row:row+size, col:col+size]
            expected[:, :, row:row+size, col:col+size] = block.mean((-2, -1), keepdim=True)
    torch.testing.assert_close(model(x), x + expected)
