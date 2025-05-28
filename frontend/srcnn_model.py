
import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, kernel_size=1)
        self.layer3 = nn.Conv2d(32, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)