"""
RCAN-style model with Channel Attention
Residual Channel Attention Network for better PSNR
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block"""
    def __init__(self, channels, reduction=16, res_scale=1.0):
        super(RCAB, self).__init__()
        self.res_scale = res_scale
        
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.attention = ChannelAttention(channels, reduction)
        
    def forward(self, x):
        res = self.body(x)
        res = self.attention(res)
        res = res * self.res_scale
        return x + res


class ResidualGroup(nn.Module):
    """Residual Group with multiple RCAB blocks"""
    def __init__(self, channels, num_blocks, reduction=16):
        super(ResidualGroup, self).__init__()
        
        modules = [RCAB(channels, reduction) for _ in range(num_blocks)]
        modules.append(nn.Conv2d(channels, channels, 3, padding=1))
        self.body = nn.Sequential(*modules)
        
    def forward(self, x):
        res = self.body(x)
        return res + x


class RCAN(nn.Module):
    """Residual Channel Attention Network"""
    def __init__(self, num_channels=64, num_groups=4, num_blocks=4, reduction=16, scale=3):
        super(RCAN, self).__init__()
        
        self.scale = scale
        
        # Shallow feature extraction
        self.head = nn.Conv2d(3, num_channels, 3, padding=1)
        
        # Deep feature extraction (Residual Groups)
        self.body = nn.ModuleList([
            ResidualGroup(num_channels, num_blocks, reduction) 
            for _ in range(num_groups)
        ])
        self.body_tail = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        
        # Upsampling
        self.tail = nn.Sequential(
            nn.Conv2d(num_channels, 3 * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale)
        )
        
    def forward(self, x):
        # Shallow features
        x = self.head(x)
        shallow = x
        
        # Deep features with residual groups
        for group in self.body:
            x = group(x)
        x = self.body_tail(x)
        x = x + shallow  # Global residual
        
        # Upsample
        x = self.tail(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test configurations to stay under 10M params
    configs = [
        {'num_channels': 64, 'num_groups': 5, 'num_blocks': 4, 'reduction': 16},
        {'num_channels': 80, 'num_groups': 4, 'num_blocks': 4, 'reduction': 16},
        {'num_channels': 64, 'num_groups': 6, 'num_blocks': 3, 'reduction': 8},
    ]
    
    for i, config in enumerate(configs):
        model = RCAN(**config)
        params = model.count_parameters()
        x = torch.randn(1, 3, 256, 256)
        y = model(x)
        print(f"Config {i+1}: {params:,} params, output: {y.shape}")
        if params > 10_000_000:
            print(f"  ⚠️  TOO LARGE!")
        else:
            print(f"  ✓ OK")

