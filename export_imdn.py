"""
Export trained IMDN model to ONNX for MLA100 deployment.
"""

import argparse
import torch

from model_imdn import IMDN


def parse_args():
    parser = argparse.ArgumentParser(description="Export IMDN to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--output", default="imdn.onnx", help="Output ONNX file path")
    parser.add_argument("--num_features", type=int, default=48)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--depthwise", action="store_true", help="Use depthwise variant")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IMDN(
        num_features=args.num_features,
        num_blocks=args.num_blocks,
        upscale=args.scale,
        depthwise_blocks=args.depthwise,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 3, 256, 256, device=device)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()

