"""
Utility functions for Super Resolution
Includes PSNR calculation matching instructor's evaluation protocol
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def rgb_to_ycbcr(img):
    """Convert RGB to YCbCr color space
    
    Args:
        img: Tensor of shape (B, 3, H, W) or (3, H, W) in range [0, 1]
    
    Returns:
        Y channel tensor of same shape minus channel dim
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    # Convert to [0, 255]
    img = img * 255.0
    
    # RGB to YCbCr conversion matrix
    r = img[:, 0, :, :]
    g = img[:, 1, :, :]
    b = img[:, 2, :, :]
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    
    if squeeze:
        y = y.squeeze(0)
    
    return y


def shave_border(img, border=4):
    """Remove border pixels from image
    
    Args:
        img: Tensor of shape (H, W) or (B, H, W)
        border: Number of pixels to remove from each side
    
    Returns:
        Shaved image
    """
    if img.dim() == 2:
        return img[border:-border, border:-border]
    else:
        return img[:, border:-border, border:-border]


def calculate_psnr(sr, hr, use_y_channel=True, shave=4):
    """Calculate PSNR between SR and HR images
    
    Args:
        sr: Super-resolved image tensor (B, 3, H, W) or (3, H, W) in range [0, 1]
        hr: High-resolution image tensor (B, 3, H, W) or (3, H, W) in range [0, 1]
        use_y_channel: Whether to compute PSNR on Y channel only
        shave: Number of border pixels to ignore
    
    Returns:
        PSNR value in dB
    """
    # Ensure tensors are on CPU and detached
    sr = sr.detach().cpu()
    hr = hr.detach().cpu()
    
    # Clamp to [0, 1]
    sr = torch.clamp(sr, 0, 1)
    hr = torch.clamp(hr, 0, 1)
    
    # Convert to Y channel if requested
    if use_y_channel:
        sr = rgb_to_ycbcr(sr)
        hr = rgb_to_ycbcr(hr)
    else:
        # Convert to [0, 255]
        sr = sr * 255.0
        hr = hr * 255.0
    
    # Shave borders
    if shave > 0:
        sr = shave_border(sr, shave)
        hr = shave_border(hr, shave)
    
    # Calculate MSE
    mse = torch.mean((sr - hr) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 10 * torch.log10(255.0 ** 2 / mse)
    
    return psnr.item()


def calculate_psnr_batch(sr_batch, hr_batch, use_y_channel=True, shave=4):
    """Calculate average PSNR for a batch of images
    
    Args:
        sr_batch: Batch of SR images (B, 3, H, W) in range [0, 1]
        hr_batch: Batch of HR images (B, 3, H, W) in range [0, 1]
        use_y_channel: Whether to compute PSNR on Y channel only
        shave: Number of border pixels to ignore
    
    Returns:
        Average PSNR value in dB
    """
    psnr_sum = 0.0
    batch_size = sr_batch.shape[0]
    
    for i in range(batch_size):
        psnr = calculate_psnr(sr_batch[i], hr_batch[i], use_y_channel, shave)
        psnr_sum += psnr
    
    return psnr_sum / batch_size


def save_checkpoint(model, optimizer, epoch, psnr, path='checkpoint.pth'):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'psnr': psnr,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    psnr = checkpoint['psnr']
    print(f"Checkpoint loaded from {path} (Epoch {epoch}, PSNR {psnr:.2f} dB)")
    return epoch, psnr


if __name__ == "__main__":
    # Test PSNR calculation
    print("Testing PSNR calculation...")
    
    # Create dummy images
    hr = torch.rand(1, 3, 768, 768)
    sr = hr + torch.randn_like(hr) * 0.01  # Add small noise
    
    psnr = calculate_psnr(sr[0], hr[0], use_y_channel=True, shave=4)
    print(f"PSNR (with noise): {psnr:.2f} dB")
    
    # Test with identical images
    psnr_perfect = calculate_psnr(hr[0], hr[0], use_y_channel=True, shave=4)
    print(f"PSNR (perfect): {psnr_perfect:.2f} dB")
    
    # Test batch calculation
    psnr_batch = calculate_psnr_batch(sr, hr, use_y_channel=True, shave=4)
    print(f"PSNR (batch): {psnr_batch:.2f} dB")
    
    print("PSNR test passed!")

