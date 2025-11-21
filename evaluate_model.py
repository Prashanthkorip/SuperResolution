"""
Evaluate the trained super resolution model on validation set
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EDSR
from dataset import SRDataset
from utils import calculate_psnr


def main():
    print("=" * 80)
    print("Super Resolution Model Evaluation")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = EDSR(num_channels=96, num_blocks=16).to(device)
    checkpoint = torch.load('checkpoints_final/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint PSNR: {checkpoint['psnr']:.2f} dB")
    
    # Create validation dataset
    val_dataset = SRDataset(
        lr_dir='LR_val/val',
        hr_dir='HR_val/val',
        augment=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    print(f"\nValidation set: {len(val_dataset)} images")
    print("\nEvaluating...")
    
    # Evaluate
    psnr_list = []
    
    with torch.no_grad():
        for i, (lr, hr) in enumerate(tqdm(val_loader)):
            lr = lr.to(device)
            hr = hr.to(device)
            
            # Forward pass
            sr = model(lr)
            
            # Calculate PSNR (Y channel, shave 4 pixels)
            psnr = calculate_psnr(sr[0], hr[0], use_y_channel=True, shave=4)
            psnr_list.append(psnr)
    
    # Statistics
    avg_psnr = sum(psnr_list) / len(psnr_list)
    max_psnr = max(psnr_list)
    min_psnr = min(psnr_list)
    
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Max PSNR: {max_psnr:.2f} dB")
    print(f"Min PSNR: {min_psnr:.2f} dB")
    print(f"Std Dev: {torch.std(torch.tensor(psnr_list)):.2f} dB")
    print("=" * 80)
    
    # Compare with baseline
    print("\nComparison with bicubic baseline:")
    print(f"Bicubic baseline: 21.13 dB")
    print(f"Our model: {avg_psnr:.2f} dB")
    print(f"Improvement: +{avg_psnr - 21.13:.2f} dB")
    print("=" * 80)


if __name__ == "__main__":
    main()

