"""
Dataset for Super Resolution
Loads LR-HR pairs and applies augmentations
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random


class SRDataset(Dataset):
    """Super Resolution Dataset"""
    def __init__(self, lr_dir, hr_dir, augment=False, hr_size=768):
        """
        Args:
            lr_dir: Directory containing LR images (256x256)
            hr_dir: Directory containing HR images (256x256, will be upsampled to hr_size)
            augment: Whether to apply data augmentation
            hr_size: Target HR size (default 768 for 3x upscaling)
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.augment = augment
        self.hr_size = hr_size
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])
        
        print(f"Loaded {len(self.image_files)} image pairs from {lr_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load images
        lr_path = os.path.join(self.lr_dir, img_name)
        hr_path = os.path.join(self.hr_dir, img_name)
        
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Upsample HR to target size (768x768) using bicubic interpolation
        hr_img = TF.resize(hr_img, [self.hr_size, self.hr_size], 
                          interpolation=Image.BICUBIC)
        
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
    train_dataset = SRDataset(
        lr_dir='LR_train/train',
        hr_dir='HR_train/train',
        augment=True
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    lr, hr = train_dataset[0]
    print(f"LR shape: {lr.shape}, range: [{lr.min():.3f}, {lr.max():.3f}]")
    print(f"HR shape: {hr.shape}, range: [{hr.min():.3f}, {hr.max():.3f}]")
    
    assert lr.shape == (3, 256, 256), f"Expected LR shape (3, 256, 256), got {lr.shape}"
    assert hr.shape == (3, 768, 768), f"Expected HR shape (3, 768, 768), got {hr.shape}"
    print("Dataset test passed!")

