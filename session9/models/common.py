import torch
import torch.nn as nn
from torchinfo import summary


class BaseNet(nn.Module):
    def summarize(self, device: torch.device, input_size: tuple = (1, 3, 32, 32)):
        print(summary(self.to(device), input_size=input_size))


# depthwise separable convolution
class DepthwiseSeparable(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super().__init__()

        self.depthwise = nn.Conv2d(
            nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias
        )
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
