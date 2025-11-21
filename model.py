"""
EDSR-style Super Resolution Model
- No BatchNorm (for NPU compatibility)
- < 10M parameters
- 3x upscaling (256x256 -> 768x768)
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block without BatchNorm"""
    def __init__(self, channels, res_scale=0.1):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale
        out = out + residual
        return out


class EDSR(nn.Module):
    """EDSR-style model for 3x super resolution"""
    def __init__(self, num_channels=64, num_blocks=8, scale_factor=3):
        super(EDSR, self).__init__()
        
        self.scale_factor = scale_factor
        
        # Input convolution
        self.input_conv = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])
        
        # Global skip connection conv
        self.global_skip_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        
        # Upsampling
        self.upsample_conv = nn.Conv2d(num_channels, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
    def forward(self, x):
        # Input conv
        x = self.input_conv(x)
        skip = x
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Global skip connection
        x = self.global_skip_conv(x)
        x = x + skip
        
        # Upsample to 3x
        x = self.upsample_conv(x)
        x = self.pixel_shuffle(x)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = EDSR(num_channels=64, num_blocks=8)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (1, 3, 768, 768), f"Expected (1, 3, 768, 768), got {y.shape}"
    print("Model test passed!")

