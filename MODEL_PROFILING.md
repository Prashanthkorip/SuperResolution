# Model Profiling Results

Complete profiling metrics for all three models using PyTorch profiler.

## Summary Table

| Model | Parameters | Size (GB) | GFLOPs | PSNR (dB) |
|-------|-----------|-----------|--------|-----------|
| EDSR Small (64ch, 12blocks) | 0.94M | 0.003504 | 123.166 | 23.40 |
| EDSR Medium (96ch, 16blocks) | 2.77M | 0.010306 | 362.369 | 23.50 |
| RCAN Large (128ch, 6 groups, 4 blocks) | 8.20M | 0.030563 | 1068.197 | 24.13 |

## Detailed Metrics

### EDSR Small (64ch, 12blocks)
- **Total Parameters**: 940,571 (0.94M)
- **Model Size**: 0.003504 GB
- **FLOPs**: 123,165,736,960 (123.166 GFLOPs)
- **PSNR**: 23.40 dB
- **Input Shape**: (1, 3, 256, 256)
- **Output Shape**: (1, 3, 768, 768)
- **NPU Compatible**: ✓ (under 10M params)

### EDSR Medium (96ch, 16blocks)
- **Total Parameters**: 2,766,363 (2.77M)
- **Model Size**: 0.010306 GB
- **FLOPs**: 362,368,991,232 (362.369 GFLOPs)
- **PSNR**: 23.50 dB
- **Input Shape**: (1, 3, 256, 256)
- **Output Shape**: (1, 3, 768, 768)
- **NPU Compatible**: ✓ (under 10M params)

### RCAN Large (128ch, 6 groups, 4 blocks)
- **Total Parameters**: 8,204,251 (8.20M)
- **Model Size**: 0.030563 GB
- **FLOPs**: 1,068,197,052,416 (1068.197 GFLOPs)
- **PSNR**: 24.13 dB
- **Input Shape**: (1, 3, 256, 256)
- **Output Shape**: (1, 3, 768, 768)
- **NPU Compatible**: ✓ (under 10M params)

## Analysis

### Parameter Efficiency
- **EDSR Small**: Most parameter-efficient (0.94M params) with 23.40 dB PSNR
- **EDSR Medium**: 2.94x more parameters than Small, +0.10 dB improvement
- **RCAN Large**: 8.72x more parameters than Small, +0.73 dB improvement

### FLOP Efficiency
- **EDSR Small**: Most FLOP-efficient (123.17 GFLOPs)
- **EDSR Medium**: 2.94x more FLOPs than Small
- **RCAN Large**: 8.67x more FLOPs than Small

### Performance vs Efficiency Trade-off
- **Best PSNR**: RCAN Large (24.13 dB) - exceeds 24 dB target
- **Best Efficiency**: EDSR Small (0.94M params, 123 GFLOPs)
- **Balanced**: EDSR Medium (2.77M params, 23.50 dB)

## Notes

- All models are NPU-compatible (under 10M parameter limit)
- Model size calculated assuming float32 (4 bytes per parameter)
- FLOPs measured using PyTorch profiler on CPU
- PSNR measured on validation set with Y-channel and 4-pixel shave

## Usage

To regenerate these metrics, run:
```bash
source .venv/bin/activate
python profile_models.py
```

