"""
IMDN-style lightweight super-resolution model
- Fewer than 1M parameters (default config)
- 3x upscaling for 256x256 -> 768x768
"""

import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, depthwise=False):
    """
    3x3 convolution helper.
    When depthwise=True, use DW + pointwise to reduce FLOPs while preserving expressiveness.
    """
    if not depthwise:
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
    )


class IMDBlock(nn.Module):
    """Information multi-distillation block from IMDN (Tao et al., 2019)"""

    def __init__(self, channels, distill_rate=0.25, depthwise=False):
        super().__init__()
        self.distilled_channels = int(channels * distill_rate)
        self.remaining_channels = channels - self.distilled_channels
        self.depthwise = depthwise

        self.relu = nn.LeakyReLU(0.05, inplace=True)
        self.conv1 = conv3x3(channels, channels, depthwise=depthwise)
        self.conv2 = conv3x3(self.remaining_channels, channels, depthwise=depthwise)
        self.conv3 = conv3x3(self.remaining_channels, channels, depthwise=depthwise)
        self.conv4 = conv3x3(self.remaining_channels, channels, depthwise=depthwise)
        self.conv_out = nn.Conv2d(
            self.distilled_channels * 4, channels, kernel_size=1, bias=True
        )

    def forward(self, x):
        distilled = []

        out1 = self.relu(self.conv1(x))
        d1, r1 = torch.split(out1, [self.distilled_channels, self.remaining_channels], dim=1)
        distilled.append(d1)

        out2 = self.relu(self.conv2(r1))
        d2, r2 = torch.split(out2, [self.distilled_channels, self.remaining_channels], dim=1)
        distilled.append(d2)

        out3 = self.relu(self.conv3(r2))
        d3, r3 = torch.split(out3, [self.distilled_channels, self.remaining_channels], dim=1)
        distilled.append(d3)

        out4 = self.relu(self.conv4(r3))
        d4 = out4[:, : self.distilled_channels, :, :]
        distilled.append(d4)

        out = torch.cat(distilled, dim=1)
        out = self.conv_out(out)
        return out + x


class IMDN(nn.Module):
    """Information Multi-Distillation Network"""

    def __init__(self, num_features=48, num_blocks=8, upscale=3, depthwise_blocks=False):
        super().__init__()
        self.upscale = upscale

        self.head = conv3x3(3, num_features, depthwise=depthwise_blocks)
        self.body = nn.ModuleList(
            [IMDBlock(num_features, depthwise=depthwise_blocks) for _ in range(num_blocks)]
        )
        self.body_tail = conv3x3(num_features, num_features, depthwise=depthwise_blocks)

        up_channels = 3 * (upscale ** 2)
        self.tail = nn.Sequential(
            conv3x3(num_features, up_channels, depthwise=False),
            nn.PixelShuffle(upscale),
        )

    def forward(self, x):
        fea = self.head(x)
        res = fea
        for block in self.body:
            res = block(res)
        res = self.body_tail(res) + fea
        return self.tail(res)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = IMDN(num_features=48, num_blocks=8, upscale=3, depthwise_blocks=True)
    params = model.count_parameters()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Parameters: {params:,}")
    print(f"Output shape: {y.shape}")

