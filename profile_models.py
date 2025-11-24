"""
Profile all models to get FLOPs, parameters, and other metrics
Based on pytorch_model_analyzer_student.ipynb
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from model import EDSR
from model_rcan import RCAN


def count_parameters(model):
    """Count total parameters"""
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_count


def get_model_size_gb(param_count):
    """Calculate model size in GB (assuming float32 = 4 bytes per parameter)"""
    return param_count * 4 / (1024 * 1024 * 1024)


def profile_model(model, model_name, dummy_input, device='cpu'):
    """
    Profile a model to get FLOPs and other metrics
    
    Args:
        model: PyTorch model
        model_name: Name of the model
        dummy_input: Dummy input tensor
        device: Device to run on ('cpu' or 'cuda')
    """
    print("=" * 80)
    print(f"Profiling: {model_name}")
    print("=" * 80)
    
    # Move model and input to device
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    model.eval()
    
    # Count parameters
    param_count = count_parameters(model)
    param_count_m = param_count / 1_000_000
    model_size_gb = get_model_size_gb(param_count)
    
    print(f"\nTotal Parameters: {param_count:,}")
    print(f"Model Size: {param_count_m:.2f} M parameters")
    print(f"Model Size: {model_size_gb:.6f} GB")
    
    # Forward pass check
    print(f"\nInput shape: {dummy_input.shape}")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # FLOP profiling - use CPU for accurate FLOP counting
    print("\nProfiling FLOPs...")
    # Note: FLOP counting works better on CPU, so we'll use CPU for profiling
    # even if the model was trained on CUDA
    model_cpu = model.cpu()
    dummy_input_cpu = dummy_input.cpu()
    
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with torch.no_grad():
            model_cpu(dummy_input_cpu)
    
    events = prof.events()
    
    # Total FLOPs
    total_flops = sum(e.flops for e in events if e.flops is not None)
    gflops = total_flops / 1e9
    
    print(f"\nTotal FLOPs: {total_flops:,}")
    print(f"GFLOPs: {gflops:.3f}")
    
    # Top 10 most expensive operations
    flop_events = [(e.flops, e.key) for e in events if e.flops is not None]
    flop_events.sort(reverse=True)
    
    print("\nTop 10 most expensive operations:")
    for i, (flops, op) in enumerate(flop_events[:10]):
        if flops is not None and total_flops > 0:
            pct = (flops / total_flops) * 100
            print(f"{i+1:2d}. {op[:45]:45s} | {flops:>15,} FLOPs ({pct:5.1f}%)")
    
    # Layer-by-layer parameter breakdown
    print("\nLayer parameter breakdown:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.PixelShuffle)):
            layer_params = sum(p.numel() for p in module.parameters())
            if layer_params > 0:
                print(f"{name:35s} | {layer_params:>12,} params")
    
    return {
        'name': model_name,
        'parameters': param_count,
        'parameters_m': param_count_m,
        'size_gb': model_size_gb,
        'flops': total_flops,
        'gflops': gflops,
        'input_shape': tuple(dummy_input.shape),
        'output_shape': tuple(output.shape)
    }


def main():
    print("=" * 80)
    print("Model Profiling - FLOPs, Parameters, and Metrics")
    print("=" * 80)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Dummy input for super resolution (256x256 -> 768x768)
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Define all models to profile
    models_to_profile = [
        {
            'name': 'EDSR Small (64ch, 12blocks)',
            'model': EDSR(num_channels=64, num_blocks=12),
            'checkpoint': 'checkpoints/best_model.pth'
        },
        {
            'name': 'EDSR Medium (96ch, 16blocks)',
            'model': EDSR(num_channels=96, num_blocks=16),
            'checkpoint': 'checkpoints_final/best_model.pth'
        },
        {
            'name': 'RCAN Large (128ch, 6 groups, 4 blocks)',
            'model': RCAN(num_channels=128, num_groups=6, num_blocks=4, reduction=16),
            'checkpoint': 'checkpoints_rcan_large/best_model.pth'
        }
    ]
    
    results = []
    
    for model_config in models_to_profile:
        try:
            # Load checkpoint if available
            model = model_config['model']
            checkpoint_path = model_config['checkpoint']
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                psnr = checkpoint.get('psnr', None)
                if psnr:
                    print(f"\nLoaded checkpoint: {checkpoint_path}")
                    print(f"Validation PSNR: {psnr:.2f} dB")
            except Exception as e:
                print(f"\n⚠ Could not load checkpoint {checkpoint_path}: {e}")
                print("Using untrained model for profiling")
            
            # Profile model
            result = profile_model(
                model,
                model_config['name'],
                dummy_input,
                device
            )
            
            # Add PSNR if available
            if 'psnr' in locals():
                result['psnr'] = psnr
            
            results.append(result)
            print("\n")
            
        except Exception as e:
            print(f"\n✗ Error profiling {model_config['name']}: {e}")
            print("\n")
    
    # Summary table
    print("=" * 80)
    print("SUMMARY - All Models")
    print("=" * 80)
    print(f"{'Model':<45} {'Params (M)':<12} {'Size (GB)':<12} {'GFLOPs':<12} {'PSNR':<10}")
    print("-" * 80)
    
    for r in results:
        psnr_str = f"{r.get('psnr', 0):.2f} dB" if r.get('psnr') else "N/A"
        print(f"{r['name']:<45} {r['parameters_m']:>6.2f}M     {r['size_gb']:>10.6f}  {r['gflops']:>10.3f}  {psnr_str:<10}")
    
    print("=" * 80)
    
    # Detailed comparison
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Parameters: {r['parameters']:,} ({r['parameters_m']:.2f}M)")
        print(f"  Model Size: {r['size_gb']:.6f} GB")
        print(f"  FLOPs: {r['flops']:,} ({r['gflops']:.3f} GFLOPs)")
        if r.get('psnr'):
            print(f"  PSNR: {r['psnr']:.2f} dB")
        print(f"  Input: {r['input_shape']}")
        print(f"  Output: {r['output_shape']}")
    
    print("\n" + "=" * 80)
    print("Profiling Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

