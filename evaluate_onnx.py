"""
Evaluate ONNX model on validation/test dataset
This script can be used to test the exported ONNX model locally
The professor likely has a similar script to evaluate all submitted models
"""

import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from tqdm import tqdm
from utils import calculate_psnr


def load_image(image_path):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32)
    # Convert HWC to CHW and normalize to [0, 1]
    img = img.transpose(2, 0, 1) / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def save_image(image_array, output_path):
    """Save image from numpy array"""
    # Remove batch dimension and convert CHW to HWC
    img = image_array[0].transpose(1, 2, 0)
    # Denormalize from [0, 1] to [0, 255]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)


def evaluate_onnx(onnx_path, lr_dir, hr_dir, output_dir=None, use_y_channel=True, shave=4):
    """
    Evaluate ONNX model on dataset
    
    Args:
        onnx_path: Path to ONNX model file
        lr_dir: Directory containing low-resolution images
        hr_dir: Directory containing high-resolution ground truth images
        output_dir: Optional directory to save super-resolved images
        use_y_channel: Calculate PSNR on Y channel only
        shave: Pixels to shave from borders for PSNR calculation
    """
    print("=" * 80)
    print("ONNX Model Evaluation")
    print("=" * 80)
    
    # Load ONNX model
    print(f"\nLoading ONNX model: {onnx_path}")
    try:
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
        print(f"✓ Model loaded successfully")
        print(f"  Providers: {session.get_providers()}")
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        print(f"  Input: {input_name} {input_shape}")
        print(f"  Output: {output_name} {output_shape}")
        
    except Exception as e:
        print(f"✗ Error loading ONNX model: {e}")
        return None
    
    # Get image files
    lr_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(lr_files) != len(hr_files):
        print(f"⚠ Warning: Mismatch in file counts (LR: {len(lr_files)}, HR: {len(hr_files)})")
    
    print(f"\nFound {len(lr_files)} image pairs")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Evaluate
    print("\nEvaluating...")
    psnr_list = []
    
    for i, lr_file in enumerate(tqdm(lr_files, desc="Processing")):
        # Find corresponding HR file
        hr_file = hr_files[i] if i < len(hr_files) else None
        if hr_file is None:
            continue
        
        lr_path = os.path.join(lr_dir, lr_file)
        hr_path = os.path.join(hr_dir, hr_file)
        
        # Load images
        try:
            lr_img = load_image(lr_path)
            hr_img = load_image(hr_path)
        except Exception as e:
            print(f"\n⚠ Error loading {lr_file}: {e}")
            continue
        
        # Run inference
        try:
            outputs = session.run([output_name], {input_name: lr_img})
            sr_img = outputs[0]
        except Exception as e:
            print(f"\n⚠ Error during inference for {lr_file}: {e}")
            continue
        
        # Save output if requested
        if output_dir:
            output_path = os.path.join(output_dir, lr_file)
            save_image(sr_img, output_path)
        
        # Calculate PSNR
        if hr_file:
            try:
                # Convert to torch tensors for PSNR calculation
                sr_tensor = torch.from_numpy(sr_img[0]).float()
                hr_tensor = torch.from_numpy(hr_img[0]).float()
                
                psnr = calculate_psnr(sr_tensor, hr_tensor, use_y_channel=use_y_channel, shave=shave)
                psnr_list.append(psnr)
            except Exception as e:
                print(f"\n⚠ Error calculating PSNR for {lr_file}: {e}")
    
    # Statistics
    if psnr_list:
        avg_psnr = sum(psnr_list) / len(psnr_list)
        max_psnr = max(psnr_list)
        min_psnr = min(psnr_list)
        std_psnr = torch.std(torch.tensor(psnr_list)).item()
        
        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Max PSNR: {max_psnr:.2f} dB")
        print(f"Min PSNR: {min_psnr:.2f} dB")
        print(f"Std Dev: {std_psnr:.2f} dB")
        print(f"Number of images: {len(psnr_list)}")
        print("=" * 80)
        
        # Compare with baseline
        print("\nComparison with bicubic baseline:")
        print(f"Bicubic baseline: 21.13 dB")
        print(f"ONNX model: {avg_psnr:.2f} dB")
        print(f"Improvement: +{avg_psnr - 21.13:.2f} dB")
        print("=" * 80)
        
        return {
            'avg_psnr': avg_psnr,
            'max_psnr': max_psnr,
            'min_psnr': min_psnr,
            'std_psnr': std_psnr,
            'num_images': len(psnr_list)
        }
    else:
        print("\n⚠ No valid PSNR calculations completed")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ONNX model on dataset')
    parser.add_argument('--onnx', type=str, default='rcan_large.onnx',
                        help='Path to ONNX model file')
    parser.add_argument('--lr_dir', type=str, default='LR_val/val',
                        help='Directory containing low-resolution images')
    parser.add_argument('--hr_dir', type=str, default='HR_val/val',
                        help='Directory containing high-resolution ground truth images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Optional directory to save super-resolved images')
    parser.add_argument('--use_y_channel', action='store_true', default=True,
                        help='Calculate PSNR on Y channel only')
    parser.add_argument('--shave', type=int, default=4,
                        help='Pixels to shave from borders for PSNR')
    
    args = parser.parse_args()
    
    evaluate_onnx(
        args.onnx,
        args.lr_dir,
        args.hr_dir,
        args.output_dir,
        args.use_y_channel,
        args.shave
    )


if __name__ == "__main__":
    main()

