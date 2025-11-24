"""
Lightweight training script for IMDN-style model
- Focus on low parameter / low FLOP deployment
- Uses patch-based training (64 -> 192)
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_imdn import IMDN
from dataset_patch import SRDatasetPatch
from utils import calculate_psnr_batch, save_checkpoint


def validate(model, loader, device):
    model.eval()
    total_psnr, batches = 0.0, 0
    with torch.no_grad():
        for lr, hr in tqdm(loader, desc="Validating", leave=False):
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)
            total_psnr += calculate_psnr_batch(sr, hr, use_y_channel=True, shave=4)
            batches += 1
    return total_psnr / max(batches, 1)


def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    for lr, hr in tqdm(loader, desc="Training", leave=False):
        lr = lr.to(device)
        hr = hr.to(device)
        with torch.cuda.amp.autocast():
            sr = model(lr)
            loss = criterion(sr, hr)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(loader)


def main():
    config = {
        "num_features": 48,
        "num_blocks": 8,
        "scale": 3,
        "batch_size": 24,
        "num_epochs": 400,
        "lr": 2e-4,
        "weight_decay": 0.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "checkpoints_imdn",
        "hr_patch": 192,
        "use_depthwise": True,
    }

    os.makedirs(config["save_dir"], exist_ok=True)

    print("=" * 80)
    print("Training IMDN-Light (low param / low FLOP)")
    print("=" * 80)
    print(f"Device: {config['device']}")

    train_dataset = SRDatasetPatch(
        "LR_train/train",
        "HR_train/train",
        hr_patch_size=config["hr_patch"],
        scale=config["scale"],
        augment=True,
    )
    val_dataset = SRDatasetPatch(
        "LR_val/val",
        "HR_val/val",
        hr_patch_size=config["hr_patch"],
        scale=config["scale"],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = IMDN(
        num_features=config["num_features"],
        num_blocks=config["num_blocks"],
        upscale=config["scale"],
        depthwise_blocks=config["use_depthwise"],
    ).to(config["device"])

    print(f"Parameters: {model.count_parameters():,}")

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=config["lr"] * 0.1
    )
    scaler = torch.cuda.amp.GradScaler()

    best_psnr = 0.0

    for epoch in range(config["num_epochs"]):
        start = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion, config["device"], scaler)
        psnr = validate(model, val_loader, config["device"])
        scheduler.step()
        elapsed = time.time() - start

        print(
            f"Epoch {epoch+1:03d}/{config['num_epochs']} | "
            f"Loss: {loss:.4f} | Val PSNR: {psnr:.2f} dB | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | {elapsed:.1f}s"
        )

        if psnr > best_psnr:
            best_psnr = psnr
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                psnr,
                os.path.join(config["save_dir"], "best_model.pth"),
            )
            print(f"★ New best PSNR: {best_psnr:.2f} dB")

    print("=" * 80)
    print(f"Training done. Best PSNR: {best_psnr:.2f} dB")
    print("=" * 80)


if __name__ == "__main__":
    main()

