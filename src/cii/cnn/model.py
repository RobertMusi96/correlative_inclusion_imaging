"""Network: ResNet-18 (torchvision), optionally pre-trained on ImageNet.

The first layer is widened from 3 to the number of input channels: the
pre-trained RGB filters are kept, the filters for the extra channels (mask)
start at zero, so at the start the network sees the image exactly as the
pre-trained one. The last layer is replaced by one output per class.
"""
from __future__ import annotations

import torch
from torch import nn


def build_model(n_classes: int, in_channels: int = 4, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with in_channels inputs and n_classes outputs; ImageNet weights if pretrained
    (downloaded by torchvision on first use)."""
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

    old = model.conv1
    new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=False)

    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3] = old.weight

    model.conv1 = new
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model
