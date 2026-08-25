import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class DSDAM(nn.Module):
    def __init__(self, in_channels, out_channels, r=2, num_heads=4, window_size=8):
        super(DSDAM, self).__init__()
        self.r = r
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.window_size = window_size
        
        self.conv_init = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.offset_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels//2, 2*3*3, kernel_size=3, padding=1),
        )
        
        self.deform_conv = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.qkv_proj = nn.Conv2d(out_channels, out_channels*3, kernel_size=1)
        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape
        residual = self.shortcut(x)
        x = self.conv_init(x)
        H_G = H // self.r
        W_G = W // self.r
        offsets = self.offset_conv(x)
        offsets = F.interpolate(offsets, size=(H_G, W_G), mode='bilinear')
        offsets_full = F.interpolate(offsets, size=(H, W), mode='bilinear')
        x = self.deform_conv(x, offsets_full)
        
        qkv = self.qkv_proj(x).view(B, 3, self.num_heads, self.out_channels//self.num_heads, H, W).permute(1, 0, 2, 4, 5, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q.contiguous().view(B, self.num_heads, H*W, self.out_channels//self.num_heads)
        k = k.contiguous().view(B, self.num_heads, H*W, self.out_channels//self.num_heads)
        v = v.contiguous().view(B, self.num_heads, H*W, self.out_channels//self.num_heads)
        
        scale = (self.out_channels//self.num_heads)**0.5
        
        if self.window_size > 0:
            num_windows = (H // self.window_size) * (W // self.window_size)
            q = q.view(B, self.num_heads, num_windows, self.window_size**2, self.out_channels//self.num_heads)
            k = k.view(B, self.num_heads, num_windows, self.window_size**2, self.out_channels//self.num_heads)
            v = v.view(B, self.num_heads, num_windows, self.window_size**2, self.out_channels//self.num_heads)
            
            attn = (q @ k.transpose(-2, -1)) / scale
            attn = F.softmax(attn, dim=-1)
            x = (attn @ v).view(B, self.num_heads, H*W, self.out_channels//self.num_heads)
        else:
            attn = (q @ k.transpose(-2, -1)) / scale
            attn = F.softmax(attn, dim=-1)
            x = (attn @ v)
        
        x = x.permute(0, 2, 1, 3).contiguous().view(B, self.out_channels, H, W)
        x = self.out_proj(x)
        x = self.norm(x + residual)
        return self.relu(x)