import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseNet, DepthwiseSeparable

"""
n_out = (n_in +2p -f) / s + 1
r_out = r_in + (k -1 ) * j_in
j_out = j_in * s

dilation: 
keff = k + (k-1)(rate - 1)
"""

class Net(BaseNet):
    def __init__(self, drop: float = 0):
        super(Net, self).__init__()

        # Block 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), # j_in = 1 | rf = 3 |
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(drop),
            DepthwiseSeparable(32, 32, 3, padding=1, bias=False), # j_in = 1 | rf = 5 |
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(drop),
            nn.Conv2d(32, 32, 3, padding=1, stride=2, bias=False), # j_in = 2 | rf = 7 |
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(drop),
        ) # 16

        self.transition1 = nn.Sequential(
            nn.Conv2d(32, 16, 1, bias=False), # rf = 7 | j_in = 2
            nn.ReLU()
        )
        # Block 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),  # j_in = 2 | rf = 11 |
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(drop),
            DepthwiseSeparable(32, 32, 3, padding=1, bias=False), # j_in = 2 | rf = 15 |
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(drop),
            nn.Conv2d(32, 32, 3, dilation=2, bias=False), # keff = 7 | j_in = 2 | rf = 27 |
            nn.Conv2d(32, 32, 3, dilation=2, bias=False), # keff = 7 | j_in = 2 | rf = 27 |
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(drop),
        ) # 8
        self.transition2 = nn.Sequential(
            nn.Conv2d(32, 16, 1, bias=False),
            nn.ReLU()
        )

        # Block 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False), # j_in = 2 | rf = 31 |
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(drop),
            DepthwiseSeparable(32, 64, 3), # j_in = 2 | rf =  35 |
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(drop),
            DepthwiseSeparable(64, 128, 3), # j_in = 2 | rf = 39 |
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(drop),
        )

        self.transition3 = nn.Sequential(
            nn.Conv2d(128, 16, 1, bias=False),
            nn.ReLU()
        )
        # Fully connected layer
        self.out = nn.Sequential(
            DepthwiseSeparable(16, 32, 3, padding=0), # j_in = 2 | rf =  43 |
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(drop),
            DepthwiseSeparable(32, 64, 3, padding=0), # j_in = 2 | rf = 47 |
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(drop),
            nn.AvgPool2d(4),
            nn.Conv2d(
                in_channels=64, out_channels=10, kernel_size=(1, 1), bias=False
            ),  # output  RF: 28
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)

        x = self.transition3(x)
        x = self.out(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)
