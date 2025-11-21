"""
Export the best model (RCAN Large) to ONNX format for MLA100 NPU deployment
"""

import torch
import torch.onnx
from model_rcan import RCAN


def export_to_onnx(model_path, output_path, input_size=(1, 3, 256, 256)):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model_path: Path to the PyTorch checkpoint
        output_path: Path to save the ONNX model
        input_size: Input tensor size (batch, channels, height, width)
    """
    print("=" * 80)
    print("ONNX Export for MLA100 NPU")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = RCAN(num_channels=128, num_groups=6, num_blocks=4, reduction=16)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {model.count_parameters():,}")
    if 'psnr' in checkpoint:
        print(f"  Validation PSNR: {checkpoint['psnr']:.2f} dB")
    
    # Create dummy input
    dummy_input = torch.randn(*input_size)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Verify forward pass
    print("Verifying forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ Output shape: {output.shape}")
    print(f"  Expected: (1, 3, 768, 768)")
    
    if output.shape != (1, 3, 768, 768):
        raise ValueError(f"Unexpected output shape: {output.shape}")
    
    # Export to ONNX
    print(f"\nExporting to ONNX: {output_path}...")
    
    torch.onnx.export(
        model,                          # Model to export
        dummy_input,                    # Dummy input
        output_path,                    # Output file path
        export_params=True,             # Store trained parameter weights
        opset_version=11,                # ONNX opset version (11 is widely supported)
        do_constant_folding=True,       # Execute constant folding optimization
        input_names=['input'],          # Input tensor name
        output_names=['output'],        # Output tensor name
        dynamic_axes={
            'input': {0: 'batch_size'},  # Variable batch size
            'output': {0: 'batch_size'}  # Variable batch size
        },
        verbose=False
    )
    
    print(f"✓ ONNX model exported successfully!")
    print(f"  File: {output_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model validation passed!")
        
        # Print model info
        print(f"\nONNX Model Info:")
        print(f"  IR Version: {onnx_model.ir_version}")
        print(f"  Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
        print(f"  Input: {onnx_model.graph.input[0].name} {[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
        print(f"  Output: {onnx_model.graph.output[0].name} {[d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")
        
    except ImportError:
        print("⚠ onnx package not installed, skipping validation")
        print("  Install with: pip install onnx")
    except Exception as e:
        print(f"⚠ ONNX validation warning: {e}")
    
    print("\n" + "=" * 80)
    print("Export Complete!")
    print("=" * 80)
    print(f"\nModel ready for MLA100 NPU deployment:")
    print(f"  ONNX file: {output_path}")
    print(f"  Input: {input_size}")
    print(f"  Output: (batch_size, 3, 768, 768)")
    print("\nNext steps:")
    print("  1. Transfer ONNX file to MLA100 system")
    print("  2. Convert ONNX to MXQ format using Mobilint tools")
    print("  3. Deploy and test on NPU")
    print("=" * 80)


def main():
    # Best model: RCAN Large (24.13 dB, 8.2M params)
    model_path = 'checkpoints_rcan_large/best_model.pth'
    output_path = 'rcan_large.onnx'
    
    export_to_onnx(model_path, output_path)


if __name__ == "__main__":
    main()

