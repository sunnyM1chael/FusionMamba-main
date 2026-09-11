"""Mechanism checks, not evidence of learned alignment quality."""
import torch
from torchvision.ops import DeformConv2d
from DSDAM import IndependentDSDAM


def test_independent_offsets_do_not_read_other_modality():
    torch.manual_seed(42)
    model = IndependentDSDAM(4, num_heads=2, window_size=2).eval()
    a, b = torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4)
    with torch.no_grad():
        first, _, _ = model(a, b)
        second, _, _ = model(a, b + 10)
    torch.testing.assert_close(first, second)


def test_known_deformable_sampling_direction():
    layer = DeformConv2d(1, 1, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1)
    x = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    offsets = torch.zeros(1, 2, 4, 5)
    torch.testing.assert_close(layer(x, offsets), x)
    offsets[:, 1] = 1  # positive horizontal offset samples one column right
    expected = torch.zeros_like(x)
    expected[..., :-1] = x[..., 1:]
    torch.testing.assert_close(layer(x, offsets), expected)
