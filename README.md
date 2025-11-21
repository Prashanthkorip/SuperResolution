# Super Resolution Project

Single-image super resolution (3x upscaling: 256×256 → 768×768)

## Results Summary

### Model Performance

| Model | Parameters | Val PSNR | Improvement over Bicubic |
|-------|-----------|----------|--------------------------|
| Bicubic Baseline | - | 21.13 dB | - |
| EDSR (64ch, 12blocks) | 645K | 23.40 dB | +2.27 dB |
| EDSR (96ch, 16blocks) + Patches | 2.77M | 23.50 dB | +2.37 dB |
| RCAN Large | 8.2M | 24.13 dB | +3.00 dB |

### Architecture

**EDSR-style model** (Enhanced Deep Super Resolution):
- No BatchNorm (NPU compatible)
- Input convolution: 3 → C channels
- N residual blocks with skip connections
- Pixel shuffle upsampling (3x)
- L1 loss for training

### Training Configuration

**Final Model:**
- Channels: 96
- Residual blocks: 16
- Patch size: 64×64 LR → 192×192 HR
- Batch size: 16
- Optimizer: Adam (lr=1e-4, StepLR decay)
- Epochs: 300
- Mixed precision training: Yes
- Data augmentation: Random flips, rotations

## Files

### Core Files
- `model.py` - EDSR architecture implementation
- `dataset.py` - Full image dataset loader
- `dataset_patch.py` - Patch-based dataset loader
- `utils.py` - PSNR calculation and utilities
- `train_final.py` - Training script with all optimizations

### Evaluation
- `evaluate_baseline.py` - Bicubic baseline evaluation
- `evaluate_model.py` - Model evaluation script

### Checkpoints
- `checkpoints_final/best_model.pth` - EDSR model (23.50 dB)
- `checkpoints_rcan_large/best_model.pth` - RCAN Large model (24.13 dB) ⭐ Latest

## Usage

### Training
```bash
python train_final.py
```

### Evaluation
```bash
# Evaluate bicubic baseline
python evaluate_baseline.py

# Evaluate trained model
python evaluate_model.py
```

### Model Export (Future)
```bash
# Export to ONNX
python export_onnx.py

# Convert to MXQ (on Mobilint machine)
python convert_onnx_to_mxq.py
```

## Dataset Structure

```
LR_train/train/  - 258 training LR images (256×256)
HR_train/train/  - 258 training HR images (256×256, upsampled to 768×768)
LR_val/val/      - 100 validation LR images (256×256)
HR_val/val/      - 100 validation HR images (256×256, upsampled to 768×768)
```

## Key Design Decisions

1. **No BatchNorm**: Mobilint NPU doesn't support BatchNorm well
2. **Patch-based training**: Better GPU utilization and data augmentation
3. **L1 Loss**: Better for PSNR optimization than MSE
4. **Pixel Shuffle**: Efficient learnable upsampling
5. **Residual Blocks**: Easier gradient flow for deep networks

## Performance Analysis

The latest **RCAN Large model achieved 24.13 dB PSNR**, beating the bicubic baseline by 3.00 dB and exceeding the 24 dB target. The previous EDSR model achieved 23.50 dB. While both models fall short of the stretch goal (26 dB), they demonstrate clear learning and the RCAN architecture successfully meets the minimum requirement.

**Factors limiting performance:**
- Dataset size (258 train, 100 val images)
- Model capacity constraint (<10M parameters for NPU)
- Degradations in LR images (compression + noise + blur)

**Possible improvements:**
- Residual dense blocks
- Channel attention mechanisms
- Perceptual loss (trades PSNR for visual quality)
- Pre-training on larger datasets
- Ensemble methods

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
numpy>=1.20.0
tqdm>=4.60.0
```

## Model Specifications for NPU

- **Parameters**: 2,766,363 (~2.77M, well under 10M limit)
- **Operations**: Conv2d, ReLU, PixelShuffle only
- **No unsupported ops**: No BatchNorm, no complex activations
- **Static shapes**: 256×256 input → 768×768 output

