"""
Patch-based Dataset for Super Resolution
Uses random cropping for better training
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random


class SRDatasetPatch(Dataset):
    """Super Resolution Dataset with random patch cropping"""
    def __init__(self, lr_dir, hr_dir, hr_patch_size=192, scale=3, augment=False):
        """
        Args:
            lr_dir: Directory containing LR images (256x256)
            hr_dir: Directory containing HR images (256x256, will be upsampled)
            hr_patch_size: Size of HR patch to extract (e.g., 192)
            scale: Upscaling factor (3x)
            augment: Whether to apply data augmentation
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale  # 192 // 3 = 64
        self.scale = scale
        self.augment = augment
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])
        
        print(f"Loaded {len(self.image_files)} image pairs from {lr_dir}")
        print(f"LR patch size: {self.lr_patch_size}x{self.lr_patch_size}")
        print(f"HR patch size: {self.hr_patch_size}x{self.hr_patch_size}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load images
        lr_path = os.path.join(self.lr_dir, img_name)
        hr_path = os.path.join(self.hr_dir, img_name)
        
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        # First upsample HR to scale*256 = 768 using bicubic
        hr_img = TF.resize(hr_img, [256 * self.scale, 256 * self.scale], 
                          interpolation=Image.BICUBIC)
        
        # Random crop
        # LR is 256x256, we want to crop lr_patch_size from it
        # HR is 768x768, we want to crop hr_patch_size from it
        # The crops must be aligned
        
        i = random.randint(0, 256 - self.lr_patch_size)
        j = random.randint(0, 256 - self.lr_patch_size)
        
        lr_img = TF.crop(lr_img, i, j, self.lr_patch_size, self.lr_patch_size)
        
        # Corresponding HR patch
        hr_i = i * self.scale
        hr_j = j * self.scale
        hr_img = TF.crop(hr_img, hr_i, hr_j, self.hr_patch_size, self.hr_patch_size)
        
        # Apply augmentations
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                lr_img = TF.hflip(lr_img)
                hr_img = TF.hflip(hr_img)
            
            # Random vertical flip
            if random.random() > 0.5:
                lr_img = TF.vflip(lr_img)
                hr_img = TF.vflip(hr_img)
            
            # Random rotation by 90 degrees
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                lr_img = TF.rotate(lr_img, angle)
                hr_img = TF.rotate(hr_img, angle)
        
        # Convert to tensor [0, 1]
        lr_tensor = TF.to_tensor(lr_img)
        hr_tensor = TF.to_tensor(hr_img)
        
        return lr_tensor, hr_tensor


if __name__ == "__main__":
    # Test the dataset
    train_dataset = SRDatasetPatch(
        lr_dir='LR_train/train',
        hr_dir='HR_train/train',
        hr_patch_size=192,
        augment=True
    )
    
    print(f"\nDataset size: {len(train_dataset)}")
    
    lr, hr = train_dataset[0]
    print(f"LR patch shape: {lr.shape}, range: [{lr.min():.3f}, {lr.max():.3f}]")
    print(f"HR patch shape: {hr.shape}, range: [{hr.min():.3f}, {hr.max():.3f}]")
    
    assert lr.shape == (3, 64, 64), f"Expected LR shape (3, 64, 64), got {lr.shape}"
    assert hr.shape == (3, 192, 192), f"Expected HR shape (3, 192, 192), got {hr.shape}"
    print("Dataset test passed!")

