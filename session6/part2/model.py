import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        drop = 0.01  # droput value (drop 10% neurons)
        self.input_layer = nn.Sequential(
            nn.Conv2d(
                1, 8, 3, padding=1, bias=False
            ),  # input: 28x28x1 output: 28x28x8 RF:3x3
            nn.ReLU(),  # activation function relu
            nn.BatchNorm2d(8),  # Batch normalization
            nn.Dropout2d(drop),
        )
        # padding=1
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 16, 3, bias=False),  # input: 28x28x8 output: 26x26x16 RF:5x5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(drop),
            nn.Conv2d(16, 16, 3, bias=False),  # input: 26x26x16 output: 24x24x24 RF:7x7
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(drop),
        )

        # Transition Block 1x1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # input: 24x24x24 output: 12x12x24 RF:14x14
            nn.Conv2d(16, 8, 1, bias=False),  # input: 12x12x24 output: 12x12x8 RF:14x14
            nn.ReLU(),
            nn.Dropout2d(drop),
        )
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, bias=False),  # input: 12x12x8 output: 12x12x16 RF:16x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(drop),
            nn.Conv2d(
                16, 32, 3, bias=False
            ),  # input: 12x12x16 output: 10x10x32 RF:18x18
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(drop),
        )

        # Transition Block (1x1)
        self.trans2 = nn.Sequential(
            nn.Conv2d(32, 8, 1, bias=False),  # input: 10x10x32 output: 10x10x8 RF:18x18
            nn.ReLU(),
            nn.Dropout2d(drop),
        )

        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, bias=False),  # input: 10x10x8, output: 8x8x16 RF: 20x20
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(drop),
            nn.Conv2d(16, 32, 3, bias=False),  # # input: 8x8x32 output: 6x6x32 RF:22x22
            nn.ReLU(),  # activation function
            nn.BatchNorm2d(32),
            nn.Dropout2d(drop),
        )

        # GAP Layer
        self.gap = nn.Sequential(nn.AvgPool2d(4))  # Global average pooling
        # Fully Connected Layer
        self.dense = nn.Linear(
            32, 10
        )  # 32 input neurons connected with 10 output neurons

    def forward(self, x):
        x = self.input_layer(x)  # input in conv1 block
        x = self.conv1(x)  # input in conv1 block
        x = self.trans1(x)  # input in trnasition block 1
        x = self.conv2(x)  # input in conv2 block
        x = self.trans2(x)  # input in transition block 2
        x = self.conv3(x)  # input in conv3 block
        x = self.gap(x)  # global average pooling

        x = x.view(-1, 32)  # reshape 2d tensor to 1d

        x = self.dense(x)  # Linear layer
        return F.log_softmax(x, dim=1)  # final prediction


class Net2(nn.Module):
    def __init__(self, drop: float = 0.0):
        super(Net2, self).__init__()

        # drop = 0.02 # droput value (drop 10% neurons)
        self.input_layer = nn.Sequential(
            nn.Conv2d(
                1, 4, 3, padding=1, bias=False
            ),  # input: 28x28x1 output: 28x28x4 RF:3x3
            nn.ReLU(),  # activation function relu
            nn.BatchNorm2d(4),  # Batch normalization
            nn.Dropout2d(drop),
        )
        # padding=1
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 8, 3, bias=False),  # input: 28x28x4 output: 26x26x8 RF:5x5
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(drop),
            # nn.Conv2d(8, 8, 3, bias=False), # input: 28x28x16 output: 28x28x24 RF:7x7
            # nn.ReLU(),
            # nn.BatchNorm2d(8),
            # nn.Dropout2d(drop),
        )

        # Transition Block 1x1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # input: 26x26x8 output: 13x13x8 RF:14x14
            nn.Conv2d(8, 4, 1, bias=False),  # input: 13x13x8 output: 13x13x4 RF:14x14
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout2d(drop),
        )
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, 3, bias=False),  # input: 13x13x4 output: 11x11x16 RF:16x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(drop),
            nn.Conv2d(16, 32, 3, bias=False),  # input: 11x11x16 output: 9x9x32 RF:18x18
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(drop),
        )

        # Transition Block (1x1)
        self.trans2 = nn.Sequential(
            nn.Conv2d(32, 8, 1, bias=False),  # input: 9x9x32 output: 9x9x8 RF:18x18
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(drop),
        )

        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, bias=False),  # input: 9x9x8, output: 7x7x16 RF: 20x20
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(drop),
            nn.Conv2d(16, 32, 3, bias=False),  # # input: 7x7x16 output: 5x5x32 RF:22x22
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(drop),
        )

        # GAP Layer
        self.gap = nn.Sequential(nn.AvgPool2d(5))  # Global average pooling
        # Fully Connected Layer
        # Fully connected layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # output  RF: 28
        )

    def forward(self, x):
        x = self.input_layer(x)  # input in conv1 block
        x = self.conv1(x)  # input in conv1 block
        x = self.trans1(x)  # input in trnasition block 1
        x = self.conv2(x)  # input in conv2 block
        x = self.trans2(x)  # input in transition block 2
        x = self.conv3(x)  # input in conv3 block
        x = self.gap(x)  # global average pooling
        x = self.conv4(x)

        x = x.view(-1, 10)  # reshape 2d tensor to 1d
        return F.log_softmax(x, dim=1)  # final prediction


def modelsummary(
    model: nn.Module, device: torch.device, input_size: tuple = (1, 28, 28)
):
    model = model.to(device)
    summary(model, input_size=input_size)
