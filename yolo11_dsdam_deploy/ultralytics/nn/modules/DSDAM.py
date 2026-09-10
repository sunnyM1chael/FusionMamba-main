# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DSDAM (Deformable Spatial Dual-Attention Module) for infrared small target detection.

This module combines deformable convolution and windowed self-attention to enhance
small target features. A `use_dsdam` flag is provided for ablation studies:
when False, the module degenerates to a channel-projection shortcut (no enhancement),
keeping the channel dimension consistent for fair comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class DSDAM(nn.Module):
    """Deformable Spatial Dual-Attention Module.

    Args:
        in_channels (int): Input channel count.
        out_channels (int): Output channel count.
        r (int): Offset downsampling ratio.
        num_heads (int): Number of attention heads.
        window_size (int): Window size for local self-attention; 0 means global attention.
        use_dsdam (bool): Whether to enable DSDAM enhancement. When False, only the
            channel-projection shortcut is applied (for ablation).
    """

    def __init__(
        self, in_channels, out_channels, r=2, num_heads=4, window_size=8,
        use_dsdam=True, energy_guided=True
    ):
        super().__init__()
        self.use_dsdam = use_dsdam
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        if not use_dsdam:
            # Ablation: keep only the channel projection, no enhancement parameters.
            return

        self.r = r
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.window_size = window_size
        self.energy_guided = energy_guided

        self.conv_init = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, 2 * 3 * 3, kernel_size=3, padding=1),
        )
        self.deform_conv = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.qkv_proj = nn.Conv2d(out_channels, out_channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        if energy_guided:
            self.energy_projection = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        if not self.use_dsdam:
            return self.shortcut(x)

        B, C, H, W = x.shape
        residual = self.shortcut(x)
        x = self.conv_init(x)
        if self.energy_guided:
            local_background = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            energy = torch.abs(x - local_background)
            energy_map = torch.sigmoid(self.energy_projection(energy))
            self.last_energy_map = energy_map.detach()
            offset_input = x * (1.0 + energy_map)
        else:
            energy_map = None
            offset_input = x
        H_G = max(H // self.r, 1)
        W_G = max(W // self.r, 1)
        offsets = self.offset_conv(offset_input)
        offsets = F.interpolate(offsets, size=(H_G, W_G), mode="bilinear")
        offsets_full = F.interpolate(offsets, size=(H, W), mode="bilinear")
        x = self.deform_conv(x, offsets_full)

        head_dim = self.out_channels // self.num_heads
        qkv = (
            self.qkv_proj(x)
            .view(B, 3, self.num_heads, head_dim, H, W)
            .permute(1, 0, 2, 4, 5, 3)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, nh, H, W, head_dim)
        scale = head_dim ** 0.5

        if self.window_size > 0:
            # Pad H/W to multiples of window_size (e.g. P5 20x20 with ws=8 -> 24x24).
            H_pad = (self.window_size - H % self.window_size) % self.window_size
            W_pad = (self.window_size - W % self.window_size) % self.window_size
            if H_pad > 0 or W_pad > 0:
                q = F.pad(q, (0, 0, 0, W_pad, 0, H_pad))
                k = F.pad(k, (0, 0, 0, W_pad, 0, H_pad))
                v = F.pad(v, (0, 0, 0, W_pad, 0, H_pad))
            Hp, Wp = H + H_pad, W + W_pad
            nh, ws = self.num_heads, self.window_size
            nH, nW = Hp // ws, Wp // ws
            # (B, nh, Hp, Wp, head_dim) -> (B, nh, nH, ws, nW, ws, head_dim) -> (B, nh, nH*nW, ws*ws, head_dim)
            q = (
                q.view(B, nh, nH, ws, nW, ws, head_dim)
                .permute(0, 1, 2, 4, 3, 5, 6)
                .contiguous()
                .view(B, nh, nH * nW, ws * ws, head_dim)
            )
            k = (
                k.view(B, nh, nH, ws, nW, ws, head_dim)
                .permute(0, 1, 2, 4, 3, 5, 6)
                .contiguous()
                .view(B, nh, nH * nW, ws * ws, head_dim)
            )
            v = (
                v.view(B, nh, nH, ws, nW, ws, head_dim)
                .permute(0, 1, 2, 4, 3, 5, 6)
                .contiguous()
                .view(B, nh, nH * nW, ws * ws, head_dim)
            )

            attn = (q @ k.transpose(-2, -1)) / scale
            attn = F.softmax(attn, dim=-1)
            x = attn @ v  # (B, nh, num_windows, ws*ws, head_dim)

            # reshape back to (B, nh, Hp*Wp, head_dim) and crop padding
            x = (
                x.view(B, nh, nH, nW, ws, ws, head_dim)
                .permute(0, 1, 2, 4, 3, 5, 6)
                .contiguous()
                .view(B, nh, Hp * Wp, head_dim)
            )
            if H_pad > 0 or W_pad > 0:
                x = (
                    x.view(B, nh, Hp, Wp, head_dim)[:, :, :H, :W, :]
                    .contiguous()
                    .view(B, nh, H * W, head_dim)
                )
        else:
            q = q.reshape(B, self.num_heads, H * W, head_dim)
            k = k.reshape(B, self.num_heads, H * W, head_dim)
            v = v.reshape(B, self.num_heads, H * W, head_dim)
            attn = (q @ k.transpose(-2, -1)) / scale
            attn = F.softmax(attn, dim=-1)
            x = attn @ v

        # (batch, head, pixel, channel_per_head) -> NCHW. Keep each
        # channel's spatial sequence contiguous before restoring H and W.
        x = x.permute(0, 1, 3, 2).contiguous().view(B, self.out_channels, H, W)
        x = self.out_proj(x)
        if energy_map is not None:
            x = x * (1.0 + energy_map)
        x = self.norm(x + residual)
        return self.relu(x)
