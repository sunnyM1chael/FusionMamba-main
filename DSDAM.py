"""Shape-adaptive deformable attention used by the fusion network.

The original project applied one independent DSDAM pass to each modality. This
version adds a cross-modal offset predictor so deformation is conditioned on
both infrared and visible features and therefore performs explicit alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class DSDAM(nn.Module):
    """Dynamic shape-adaptive deformable attention with local window attention."""

    def __init__(self, in_channels, out_channels, r=2, num_heads=4, window_size=8):
        super().__init__()
        if out_channels % num_heads:
            raise ValueError(f"out_channels={out_channels} must be divisible by num_heads={num_heads}")
        self.r = r
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.window_size = window_size

        self.conv_init = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        hidden = max(out_channels // 2, 1)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(out_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2 * 3 * 3, kernel_size=3, padding=1),
        )
        self.deform_conv = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.qkv_proj = nn.Conv2d(out_channels, out_channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def _window_attention(self, q, k, v, height, width):
        batch, heads, _, _, head_dim = q.shape
        ws = self.window_size
        pad_h = (ws - height % ws) % ws
        pad_w = (ws - width % ws) % ws
        if pad_h or pad_w:
            q = F.pad(q, (0, 0, 0, pad_w, 0, pad_h))
            k = F.pad(k, (0, 0, 0, pad_w, 0, pad_h))
            v = F.pad(v, (0, 0, 0, pad_w, 0, pad_h))

        hp, wp = height + pad_h, width + pad_w
        nh, nw = hp // ws, wp // ws

        def partition(tensor):
            return (
                tensor.view(batch, heads, nh, ws, nw, ws, head_dim)
                .permute(0, 1, 2, 4, 3, 5, 6)
                .contiguous()
                .view(batch, heads, nh * nw, ws * ws, head_dim)
            )

        q, k, v = partition(q), partition(k), partition(v)
        attention = (q @ k.transpose(-2, -1)) / (head_dim**0.5)
        output = attention.softmax(dim=-1) @ v
        output = (
            output.view(batch, heads, nh, nw, ws, ws, head_dim)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .contiguous()
            .view(batch, heads, hp, wp, head_dim)
        )
        return output[:, :, :height, :width].contiguous().view(batch, heads, height * width, head_dim)

    def forward(self, x, offsets=None):
        """Enhance ``x``; optional offsets allow cross-modal alignment guidance."""
        batch, _, height, width = x.shape
        residual = self.shortcut(x)
        x = self.conv_init(x)

        if offsets is None:
            offsets = self.offset_conv(x)
            grid_h = max(height // self.r, 1)
            grid_w = max(width // self.r, 1)
            offsets = F.interpolate(offsets, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
            offsets = F.interpolate(offsets, size=(height, width), mode="bilinear", align_corners=False)
        elif offsets.shape[-2:] != (height, width):
            offsets = F.interpolate(offsets, size=(height, width), mode="bilinear", align_corners=False)

        x = self.deform_conv(x, offsets)
        head_dim = self.out_channels // self.num_heads
        qkv = (
            self.qkv_proj(x)
            .view(batch, 3, self.num_heads, head_dim, height, width)
            .permute(1, 0, 2, 4, 5, 3)
        )
        q, k, v = qkv.unbind(0)

        if self.window_size > 0:
            x = self._window_attention(q, k, v, height, width)
        else:
            q = q.reshape(batch, self.num_heads, height * width, head_dim)
            k = k.reshape(batch, self.num_heads, height * width, head_dim)
            v = v.reshape(batch, self.num_heads, height * width, head_dim)
            x = ((q @ k.transpose(-2, -1)) / (head_dim**0.5)).softmax(dim=-1) @ v

        # [B, heads, pixels, head_dim] -> [B, heads, head_dim, pixels].
        # Keep spatial positions intact when merging heads into channels.
        x = x.permute(0, 1, 3, 2).contiguous().view(batch, self.out_channels, height, width)
        return self.relu(self.norm(self.out_proj(x) + residual))


class CrossModalDSDAM(nn.Module):
    """Jointly predict bidirectional offsets and align IR/VIS features."""

    def __init__(self, channels, r=2, num_heads=4, window_size=8, max_offset=4.0):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.joint_projection = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.offset_head = nn.Conv2d(hidden, 2 * 2 * 3 * 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        self.max_offset = float(max_offset)
        self.enhancer = DSDAM(channels, channels, r=r, num_heads=num_heads, window_size=window_size)

    def forward(self, infrared, visible):
        joint = torch.cat(
            [infrared, visible, torch.abs(infrared - visible), infrared * visible], dim=1
        )
        offsets = torch.tanh(self.offset_head(self.joint_projection(joint))) * self.max_offset
        offset_ir, offset_vis = offsets.chunk(2, dim=1)

        # One batched call gives BatchNorm balanced statistics from both modalities.
        pair = torch.cat([infrared, visible], dim=0)
        pair_offsets = torch.cat([offset_ir, offset_vis], dim=0)
        aligned_ir, aligned_vis = self.enhancer(pair, pair_offsets).chunk(2, dim=0)
        confidence = torch.exp(-torch.mean(torch.abs(aligned_ir - aligned_vis), dim=1, keepdim=True))
        return aligned_ir, aligned_vis, confidence


class IndependentDSDAM(nn.Module):
    """Shared enhancer, but each sample predicts offsets from its own modality.

    Batched evaluation matches the joint variant's BatchNorm treatment. This is
    not two independently parameterized encoders and is not geometric alignment.
    """
    def __init__(self, channels, r=2, num_heads=4, window_size=8):
        super().__init__()
        self.enhancer = DSDAM(channels, channels, r, num_heads, window_size)

    def forward(self, infrared, visible):
        a, b = self.enhancer(torch.cat([infrared, visible], dim=0)).chunk(2, dim=0)
        return a, b, None
