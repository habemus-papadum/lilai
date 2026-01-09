"""
ResNet implementation from scratch.
Learning exercise to understand the architecture deeply.
"""

import torch
import torch.nn as nn
from typing import Type, List, Optional, Callable, Union


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet18 and ResNet34.

    Structure:
        x -> [3x3 conv -> BN -> ReLU -> 3x3 conv -> BN] -> + -> ReLU -> out
             |_____________ shortcut ___________________|

    The shortcut connection either:
    - Passes x directly (if dimensions match)
    - Uses a 1x1 conv to match dimensions (if stride > 1 or channels change)
    """

    expansion: int = 1  # Output channels = planes * expansion

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Save the input as 'identity' for the skip connection
        identity = x
        # 2. Pass through: conv1 -> bn1 -> relu -> conv2 -> bn2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # 3. If self.downsample is not None, apply it to identity
        if self.downsample is not None:
            identity = self.downsample(identity)
        # 4. Add identity to the output (the residual connection!)
        x  = x + identity
        # 5. Apply relu to the sum and return
        x = self.relu(x)
        return x