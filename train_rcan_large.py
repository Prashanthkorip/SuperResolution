"""
Train Large RCAN - Method 3: Maximum capacity under 10M params
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from model_rcan import RCAN
from dataset_patch import SRDatasetPatch
from dataset import SRDataset
from utils import calculate_psnr_batch, save_checkpoint


def validate(model, val_loader, device):
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for lr, hr in tqdm(val_loader, desc="Val", leave=False):
            sr = model(lr.to(device))
            total_psnr += calculate_psnr_batch(sr, hr.to(device), True, 4)
    return total_psnr / len(val_loader)


def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    for lr, hr in tqdm(train_loader, desc="Train", leave=False):
        with torch.cuda.amp.autocast():
            loss = criterion(model(lr.to(device)), hr.to(device))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def main():
    config = {
        'num_channels': 128,
        'num_groups': 6,
        'num_blocks': 4,
        'reduction': 16,
        'batch_size': 12,  # Reduced for larger model
        'hr_patch_size': 192,
        'num_epochs': 200,  # Faster training
        'lr': 1e-4,
        'save_dir': 'checkpoints_rcan_large',
    }
    
    print("=" * 80)
    print("Method 3: Large RCAN (8.2M parameters)")
    print("=" * 80)
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    train_dataset = SRDatasetPatch('LR_train/train', 'HR_train/train', 
                                   hr_patch_size=config['hr_patch_size'], augment=True)
    val_dataset = SRDataset('LR_val/val', 'HR_val/val', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    model = RCAN(num_channels=config['num_channels'], num_groups=config['num_groups'],
                 num_blocks=config['num_blocks'], reduction=config['reduction']).cuda()
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    scaler = torch.cuda.amp.GradScaler()
    
    best_psnr = 0.0
    
    for epoch in range(config['num_epochs']):
        start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 'cuda', scaler)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_psnr = validate(model, val_loader, 'cuda')
            elapsed = time.time() - start
            
            print(f"Epoch {epoch+1}/{config['num_epochs']} - Loss: {train_loss:.4f}, "
                  f"PSNR: {val_psnr:.2f} dB, Time: {elapsed:.1f}s")
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(model, optimizer, epoch+1, val_psnr, 
                              f"{config['save_dir']}/best_model.pth")
                print(f"★ New best: {best_psnr:.2f} dB")
                
                if val_psnr >= 26.0:
                    print(f"\n{'='*80}\n✓✓✓ GOAL: {val_psnr:.2f} dB!\n{'='*80}")
                    break
        
        scheduler.step()
    
    print(f"\n{'='*80}\nMethod 3 Complete - Best: {best_psnr:.2f} dB\n{'='*80}\n")
    return best_psnr


if __name__ == "__main__":
    main()

