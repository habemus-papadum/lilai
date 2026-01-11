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
        self.relu = nn.ReLU(inplace=True)
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


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet50, ResNet101, and ResNet152.

    Structure (1x1 -> 3x3 -> 1x1):
        x -> [1x1 conv -> BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 1x1 conv -> BN] -> + -> ReLU -> out
             |__________________________ shortcut __________________________________|

    The "bottleneck" design:
    - First 1x1 conv REDUCES channels (in_channels -> planes)
    - Middle 3x3 conv processes at reduced dimension (planes -> planes)
    - Final 1x1 conv EXPANDS channels (planes -> planes * 4)

    This is more parameter-efficient than BasicBlock for the same output channels.
    """

    expansion: int = 4  # Output channels = planes * expansion

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        
        # You need to create:
        # 1. self.conv1: 1x1 conv, in_channels -> planes, stride=1, no bias
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, stride=1, bias=False)
        # 2. self.bn1: BatchNorm2d for planes
        self.bn1 = nn.BatchNorm2d(planes)
        # 3. self.conv2: 3x3 conv, planes -> planes, stride=stride, padding=1, no bias
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # 4. self.bn2: BatchNorm2d for planes
        self.bn2 = nn.BatchNorm2d(planes)
        # 5. self.conv3: 1x1 conv, planes -> planes * self.expansion, stride=1, no bias
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        # 6. self.bn3: BatchNorm2d for planes * self.expansion
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        # 7. self.relu: ReLU (will be reused)
        self.relu = nn.ReLU(inplace=True)
        # 8. self.downsample: store the downsample parameter
        self.downsample = downsample
        # 9. self.stride: store the stride parameter
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(human): Implement the forward pass (after __init__ is done)
        #
        # Similar to BasicBlock but with 3 conv layers:
        # 1. Save identity
        identity = x
        # 2. x = relu(bn1(conv1(x)))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 3. x = relu(bn2(conv2(x)))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # 4. x = bn3(conv3(x))  -- Note: NO relu here!
        x = self.conv3(x)
        x = self.bn3(x)
        # 5. Apply downsample to identity if needed
        if self.downsample is not None:
            identity = self.downsample(identity)
        # 6. x = relu(x + identity)
        x = x + identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    """
    ResNet architecture.

    The network consists of:
    1. Initial convolution + pooling (reduces 224x224 -> 56x56)
    2. Four "layers" of stacked residual blocks
    3. Global average pooling + fully connected classifier
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
    ) -> None:
        """
        Args:
            block: The block type to use (BasicBlock or Bottleneck)
            layers: Number of blocks in each of the 4 layers, e.g., [2, 2, 2, 2] for ResNet18
            num_classes: Number of output classes (default 1000 for ImageNet)
        """
        super().__init__()
        self.in_channels = 64  # Tracks current channel count as we build layers

        # Initial convolution: 3 -> 64 channels, 224x224 -> 112x112
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # Max pooling: 112x112 -> 56x56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        assert len(layers) == 4
        # The four residual layers
        # TODO(human): Create the four layers using self._make_layer
        #
        self.layer1 = self._make_layer(block, planes=64, num_blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, planes=128, num_blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes=256, num_blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes=512, num_blocks=layers[3], stride=2)
        
        # Note: layer1 has stride=1 (no downsampling), layers 2-4 have stride=2

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        Creates a layer of stacked residual blocks.

        Args:
            block: Block type (BasicBlock or Bottleneck)
            planes: Number of planes (channels) for this layer
            num_blocks: How many blocks to stack
            stride: Stride for the FIRST block (subsequent blocks use stride=1)

        Returns:
            nn.Sequential containing all blocks for this layer
        """
        
        #
        # Key insight: Only the FIRST block may need downsampling!
        # - If stride > 1, spatial dimensions change
        # - If in_channels != planes * expansion, channel count changes
        #
        # Steps:
        # 1. Determine if we need a downsample module:
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                   nn.Conv2d(self.in_channels, planes * block.expansion, 
                             kernel_size=1, stride=stride, bias=False),
                   nn.BatchNorm2d(planes * block.expansion)
               )
        else:
               downsample = None
        #
        # 2. Create list of blocks:
        blocks = []
        blocks.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for _ in range(1, num_blocks):
            blocks.append(block(self.in_channels, planes))
        #
        # 3. Return nn.Sequential(*blocks)
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(human): Implement forward pass (after _make_layer is done)
        #
        # The data flow:
        # 1. Initial: conv1 -> bn1 -> relu -> maxpool
        
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 2. Residual layers: layer1 -> layer2 -> layer3 -> layer4
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 3. Classifier: avgpool -> flatten -> fc
        #
        # Hint: Use torch.flatten(x, 1) to flatten all dims except batch
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =============================================================================
# Factory functions to create specific ResNet variants
# =============================================================================

def resnet18(num_classes: int = 1000) -> ResNet:
    """ResNet-18: BasicBlock, [2, 2, 2, 2] blocks"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes: int = 1000) -> ResNet:
    """ResNet-34: BasicBlock, [3, 4, 6, 3] blocks"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes: int = 1000) -> ResNet:
    """ResNet-50: Bottleneck, [3, 4, 6, 3] blocks"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes: int = 1000) -> ResNet:
    """ResNet-101: Bottleneck, [3, 4, 23, 3] blocks"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes: int = 1000) -> ResNet:
    """ResNet-152: Bottleneck, [3, 8, 36, 3] blocks"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)