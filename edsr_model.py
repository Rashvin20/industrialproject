import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + 0.1 * self.block(x)  # residual scaling for stability

class EDSR(nn.Module):
    def __init__(self, scale=4, num_res_blocks=8, in_channels=3):
        super(EDSR, self).__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(64, in_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = self.tail(x + res)  # skip connection from head
        return x
