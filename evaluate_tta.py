"""
Test-Time Augmentation for PSNR boost
Geometric self-ensemble averaging
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model_rcan import RCAN
from dataset import SRDataset
from utils import calculate_psnr


def geometric_ensemble_8(model, lr, device):
    """Apply 8 geometric transformations and average"""
    def _transform(x, k, flip):
        if flip:
            x = torch.flip(x, [3])  # Horizontal flip
        x = torch.rot90(x, k, [2, 3])  # Rotate
        return x
    
    def _inverse_transform(x, k, flip):
        x = torch.rot90(x, -k, [2, 3])
        if flip:
            x = torch.flip(x, [3])
        return x
    
    outputs = []
    
    # 8 transformations: 4 rotations × 2 (with/without flip)
    for k in range(4):  # 0, 90, 180, 270 degrees
        for flip in [False, True]:
            x_trans = _transform(lr.clone(), k, flip)
            with torch.no_grad():
                out = model(x_trans)
            out = _inverse_transform(out, k, flip)
            outputs.append(out)
    
    # Average all outputs
    return torch.stack(outputs).mean(dim=0)


def main():
    print("=" * 80)
    print("Method 2: Test-Time Augmentation (TTA)")
    print("=" * 80)
    
    device = 'cuda'
    
    # Load best RCAN model
    model = RCAN(num_channels=80, num_groups=4, num_blocks=4, reduction=16).to(device)
    checkpoint = torch.load('checkpoints_rcan/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model with checkpoint PSNR: {checkpoint['psnr']:.2f} dB")
    
    # Validation dataset
    val_dataset = SRDataset('LR_val/val', 'HR_val/val', augment=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    psnr_list = []
    
    print("\nEvaluating with 8x Test-Time Augmentation...")
    for lr, hr in tqdm(val_loader):
        lr, hr = lr.to(device), hr.to(device)
        
        # Apply TTA
        sr = geometric_ensemble_8(model, lr, device)
        
        # Calculate PSNR
        psnr = calculate_psnr(sr[0], hr[0], use_y_channel=True, shave=4)
        psnr_list.append(psnr)
    
    avg_psnr = np.mean(psnr_list)
    
    print("\n" + "=" * 80)
    print(f"Without TTA: {checkpoint['psnr']:.2f} dB")
    print(f"With TTA (8x): {avg_psnr:.2f} dB")
    print(f"Improvement: +{avg_psnr - checkpoint['psnr']:.2f} dB")
    print("=" * 80)
    
    if avg_psnr >= 26.0:
        print("\n✓✓✓ GOAL ACHIEVED: 26 dB!")
    
    return avg_psnr


if __name__ == "__main__":
    best = main()

