"""
Training script for Super Resolution model V2
Larger model with more capacity to achieve PSNR >= 24 dB
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from model import EDSR
from dataset import SRDataset
from utils import calculate_psnr_batch, save_checkpoint, load_checkpoint


def validate(model, val_loader, device, use_y_channel=True, shave=4):
    """Validate the model on validation set"""
    model.eval()
    total_psnr = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for lr, hr in tqdm(val_loader, desc="Validating", leave=False):
            lr = lr.to(device)
            hr = hr.to(device)
            
            # Forward pass
            sr = model(lr)
            
            # Calculate PSNR
            psnr = calculate_psnr_batch(sr, hr, use_y_channel, shave)
            total_psnr += psnr
            num_batches += 1
    
    avg_psnr = total_psnr / num_batches
    return avg_psnr


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for lr, hr in pbar:
        lr = lr.to(device)
        hr = hr.to(device)
        
        # Forward pass
        sr = model(lr)
        loss = criterion(sr, hr)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    # Configuration - larger model
    config = {
        'num_channels': 96,  # Increased from 64
        'num_blocks': 16,    # Increased from 12
        'batch_size': 6,     # Reduced batch size due to larger model
        'num_epochs': 250,
        'lr': 1e-4,
        'lr_decay_step': 60,
        'lr_decay_gamma': 0.5,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints_v2',
        'use_y_channel': True,
        'shave': 4,
    }
    
    print("=" * 80)
    print("Super Resolution Training V2 - Larger Model")
    print("=" * 80)
    print(f"Device: {config['device']}")
    print(f"Model: EDSR with {config['num_channels']} channels, {config['num_blocks']} blocks")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Epochs: {config['num_epochs']}")
    print("=" * 80)
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = SRDataset(
        lr_dir='LR_train/train',
        hr_dir='HR_train/train',
        augment=True
    )
    
    val_dataset = SRDataset(
        lr_dir='LR_val/val',
        hr_dir='HR_val/val',
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Train set: {len(train_dataset)} images")
    print(f"Val set: {len(val_dataset)} images")
    
    # Create model
    print("\nInitializing model...")
    model = EDSR(
        num_channels=config['num_channels'],
        num_blocks=config['num_blocks']
    ).to(config['device'])
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_decay_step'],
        gamma=config['lr_decay_gamma']
    )
    
    # Training loop
    best_psnr = 0.0
    start_epoch = 0
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'], epoch + 1)
        
        # Validate
        val_psnr = validate(model, val_loader, config['device'], config['use_y_channel'], config['shave'])
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        # Print statistics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']} - "
              f"Loss: {train_loss:.4f}, Val PSNR: {val_psnr:.2f} dB, "
              f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_path = os.path.join(config['save_dir'], 'best_model.pth')
            save_checkpoint(model, optimizer, epoch + 1, val_psnr, save_path)
            print(f"★ New best PSNR: {best_psnr:.2f} dB")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, val_psnr, save_path)
        
        # Early stopping if target achieved
        if val_psnr >= 24.0:
            print(f"\n{'=' * 80}")
            print(f"✓ Target PSNR of 24 dB achieved! (Current: {val_psnr:.2f} dB)")
            print(f"{'=' * 80}")
            # Continue training to see if we can do better
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation PSNR: {best_psnr:.2f} dB")
    print("=" * 80)


if __name__ == "__main__":
    main()

