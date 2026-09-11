"""Read-only analytical probe of the deployed fusion DSDAM spatial layout."""
import json
import sys
import types
import torch

sys.path.insert(0, '/root/autodl-tmp/FusionMamba-SACAFM-14e9a93')
from DSDAM import DSDAM

torch.set_num_threads(2)
module = DSDAM(4, 4, num_heads=2, window_size=2).eval()
reference = torch.arange(24, dtype=torch.float32).reshape(1, 2, 6, 2)
def fake_attention(self, q, k, v, height, width):
    return reference.clone()
module._window_attention = types.MethodType(fake_attention, module)
captured = []
handle = module.out_proj.register_forward_pre_hook(lambda m, args: captured.append(args[0].clone()))
with torch.inference_mode():
    module(torch.zeros(1, 4, 2, 3))
handle.remove()
expected = torch.empty(1, 4, 2, 3)
for head in range(2):
    for pixel in range(6):
        for channel in range(2):
            expected[0, head * 2 + channel, pixel // 3, pixel % 3] = reference[0, head, pixel, channel]
print(json.dumps({'matches_explicit_index_reference': torch.equal(captured[0], expected),
                  'mismatched_entries': int((captured[0] != expected).sum()),
                  'total_entries': expected.numel(),
                  'correct_permutation_matches': torch.equal(reference.permute(0, 1, 3, 2).contiguous().view(1, 4, 2, 3), expected)}))
