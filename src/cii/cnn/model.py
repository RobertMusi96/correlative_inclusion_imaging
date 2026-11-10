"""Network: ResNet-18 (torchvision), optionally pre-trained on ImageNet.

The first layer is widened from 3 to the number of input channels. Every RGB
group (bright field and each lighting mode) gets a copy of the pre-trained RGB
filters divided by the number of groups, so that at the start the response
matches the pre-trained network; the filters of the mask channel start at zero.
The last layer is replaced by one output per class.
"""
from __future__ import annotations

import torch
from torch import nn


def build_model(n_classes: int, in_channels: int = 4, pretrained: bool = True, groups=([0, 1, 2],)) -> nn.Module:
    """ResNet-18 with in_channels inputs and n_classes outputs; ImageNet weights if pretrained
    (downloaded by torchvision on first use). groups: channel indices of the RGB groups."""
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

    old = model.conv1
    new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=False)

    with torch.no_grad():
        new.weight.zero_()
        for g in groups:
            new.weight[:, list(g)] = old.weight / len(groups)

    model.conv1 = new
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model
