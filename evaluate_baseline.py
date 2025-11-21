"""
Evaluate bicubic baseline for comparison
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SRDataset
from utils import calculate_psnr_batch


def main():
    print("=" * 80)
    print("Evaluating Bicubic Baseline")
    print("=" * 80)
    
    # Create validation dataset
    val_dataset = SRDataset(
        lr_dir='LR_val/val',
        hr_dir='HR_val/val',
        augment=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Validation set: {len(val_dataset)} images")
    
    # Evaluate bicubic upsampling
    total_psnr = 0.0
    num_batches = 0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with torch.no_grad():
        for lr, hr in tqdm(val_loader, desc="Evaluating bicubic"):
            lr = lr.to(device)
            hr = hr.to(device)
            
            # Bicubic upsampling (3x)
            sr = F.interpolate(lr, size=(768, 768), mode='bicubic', align_corners=False)
            
            # Calculate PSNR
            psnr = calculate_psnr_batch(sr, hr, use_y_channel=True, shave=4)
            total_psnr += psnr
            num_batches += 1
    
    avg_psnr = total_psnr / num_batches
    
    print("=" * 80)
    print(f"Bicubic baseline PSNR: {avg_psnr:.2f} dB")
    print("=" * 80)
    print(f"Target PSNR: 24.00 dB")
    print(f"Required improvement: {24.0 - avg_psnr:.2f} dB")
    print("=" * 80)


if __name__ == "__main__":
    main()

