"""
Final Training Script for Super Resolution
Target: PSNR >= 26 dB on validation set
Uses patch-based training with larger model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from model import EDSR
from dataset_patch import SRDatasetPatch
from dataset import SRDataset  # For validation on full images
from utils import calculate_psnr_batch, save_checkpoint


def validate(model, val_loader, device, use_y_channel=True, shave=4):
    """Validate the model on validation set (full images)"""
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


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for lr, hr in pbar:
        lr = lr.to(device)
        hr = hr.to(device)
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                sr = model(lr)
                loss = criterion(sr, hr)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    # Configuration
    config = {
        'num_channels': 96,      # Larger model
        'num_blocks': 16,        # More blocks
        'batch_size': 16,        # Can use larger batch with patches
        'hr_patch_size': 192,    # HR patch size
        'num_epochs': 300,       # More epochs
        'lr': 1e-4,
        'lr_decay_step': 75,     # Decay every 75 epochs
        'lr_decay_gamma': 0.5,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints_final',
        'use_y_channel': True,
        'shave': 4,
        'use_amp': True,  # Mixed precision training
        'val_freq': 5,    # Validate every 5 epochs
    }
    
    print("=" * 80)
    print("Super Resolution Training - Final Push to 26 dB")
    print("=" * 80)
    print(f"Device: {config['device']}")
    print(f"Model: EDSR with {config['num_channels']} channels, {config['num_blocks']} blocks")
    print(f"Batch size: {config['batch_size']}")
    print(f"HR Patch size: {config['hr_patch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Mixed precision: {config['use_amp']}")
    print("=" * 80)
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = SRDatasetPatch(
        lr_dir='LR_train/train',
        hr_dir='HR_train/train',
        hr_patch_size=config['hr_patch_size'],
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
        batch_size=4,  # Smaller batch for full images
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
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] and config['device'] == 'cuda' else None
    
    # Training loop
    best_psnr = 0.0
    start_epoch = 0
    patience = 40  # Early stopping patience
    no_improve_count = 0
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'], epoch + 1, scaler)
        
        # Validate periodically
        if (epoch + 1) % config['val_freq'] == 0 or epoch == 0:
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
                improvement = val_psnr - best_psnr
                best_psnr = val_psnr
                no_improve_count = 0
                save_path = os.path.join(config['save_dir'], 'best_model.pth')
                save_checkpoint(model, optimizer, epoch + 1, val_psnr, save_path)
                print(f"★ New best PSNR: {best_psnr:.2f} dB (+{improvement:.2f} dB improvement)")
                
                # Check if target achieved
                if val_psnr >= 26.0:
                    print(f"\n{'=' * 80}")
                    print(f"✓✓✓ TARGET ACHIEVED! PSNR >= 26 dB (Current: {val_psnr:.2f} dB)")
                    print(f"{'=' * 80}")
            else:
                no_improve_count += config['val_freq']
            
            # Save periodic checkpoint
            if (epoch + 1) % 50 == 0:
                save_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch + 1}.pth')
                save_checkpoint(model, optimizer, epoch + 1, val_psnr, save_path)
            
            # Early stopping check
            if no_improve_count >= patience:
                print(f"\n{'=' * 80}")
                print(f"Early stopping triggered after {no_improve_count} epochs without improvement")
                print(f"{'=' * 80}")
                break
            
            print("-" * 80)
        else:
            # Just update scheduler without validation
            scheduler.step()
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{config['num_epochs']} - "
                  f"Loss: {train_loss:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation PSNR: {best_psnr:.2f} dB")
    if best_psnr >= 26.0:
        print("✓ TARGET ACHIEVED: PSNR >= 26 dB")
    elif best_psnr >= 24.0:
        print("✓ Minimum target achieved: PSNR >= 24 dB")
    print("=" * 80)


if __name__ == "__main__":
    main()

