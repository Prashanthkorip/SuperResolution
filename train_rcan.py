"""
Train RCAN model - Method 1: Channel Attention
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
    num_batches = 0
    
    with torch.no_grad():
        for lr, hr in tqdm(val_loader, desc="Validating", leave=False):
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)
            psnr = calculate_psnr_batch(sr, hr, use_y_channel=True, shave=4)
            total_psnr += psnr
            num_batches += 1
    
    return total_psnr / num_batches


def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    
    for lr, hr in tqdm(train_loader, desc="Training", leave=False):
        lr, hr = lr.to(device), hr.to(device)
        
        with torch.cuda.amp.autocast():
            sr = model(lr)
            loss = criterion(sr, hr)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def main():
    config = {
        'num_channels': 80,
        'num_groups': 4,
        'num_blocks': 4,
        'reduction': 16,
        'batch_size': 16,
        'hr_patch_size': 192,
        'num_epochs': 300,
        'lr': 1e-4,
        'device': 'cuda',
        'save_dir': 'checkpoints_rcan',
    }
    
    print("=" * 80)
    print("Method 1: RCAN with Channel Attention")
    print("=" * 80)
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Datasets
    train_dataset = SRDatasetPatch('LR_train/train', 'HR_train/train', 
                                   hr_patch_size=config['hr_patch_size'], augment=True)
    val_dataset = SRDataset('LR_val/val', 'HR_val/val', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    # Model
    model = RCAN(num_channels=config['num_channels'], num_groups=config['num_groups'],
                 num_blocks=config['num_blocks'], reduction=config['reduction']).to(config['device'])
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    scaler = torch.cuda.amp.GradScaler()
    
    best_psnr = 0.0
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'], scaler)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_psnr = validate(model, val_loader, config['device'])
            elapsed = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{config['num_epochs']} - Loss: {train_loss:.4f}, "
                  f"Val PSNR: {val_psnr:.2f} dB, LR: {optimizer.param_groups[0]['lr']:.6f}, "
                  f"Time: {elapsed:.1f}s")
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(model, optimizer, epoch+1, val_psnr, 
                              f"{config['save_dir']}/best_model.pth")
                print(f"★ New best: {best_psnr:.2f} dB")
                
                if val_psnr >= 26.0:
                    print(f"\n{'='*80}\n✓✓✓ GOAL ACHIEVED: {val_psnr:.2f} dB\n{'='*80}")
                    break
        
        scheduler.step()
    
    print(f"\n{'='*80}")
    print(f"Method 1 Complete - Best PSNR: {best_psnr:.2f} dB")
    print(f"{'='*80}\n")
    return best_psnr


if __name__ == "__main__":
    best = main()

