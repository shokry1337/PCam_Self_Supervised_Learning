"""
models.py — ResNet-34 Backbones and Self-Supervised Wrappers

Provides:
  - ResNet34Backbone: 512-dim feature extractor (from scratch or ImageNet weights).
  - BackboneWrapper: Wrapper for Random Init and ImageNet baselines.
  - RotNetModel: ResNet-34 + 4-way rotation classification head.
  - SimCLRModel: ResNet-34 + 2-layer MLP projection head (discarded downstream).
  - BinaryClassifier: Unfrozen ResNet-34 + 2-class linear head for end-to-end fine-tuning.
"""

import copy
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet34Backbone(nn.Module):
    """ResNet-34 feature extractor outputting 512-dimensional pooled representations."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base = models.resnet34(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class BackboneWrapper(nn.Module):
    """Wrapper exposing a consistent .backbone and .get_representation() interface."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.backbone = ResNet34Backbone(pretrained=pretrained)

    @torch.no_grad()
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class RotNetModel(nn.Module):
    """ResNet-34 with a 4-way rotation classification head (0, 90, 180, 270 deg)."""

    def __init__(self):
        super().__init__()
        self.backbone = ResNet34Backbone(pretrained=False)
        self.classifier = nn.Linear(512, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.classifier(h)

    @torch.no_grad()
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SimCLRModel(nn.Module):
    """ResNet-34 with a 2-layer MLP projection head for NT-Xent contrastive training."""

    def __init__(self, proj_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.backbone = ResNet34Backbone(pretrained=False)
        self.projection_head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        z = F.normalize(self.projection_head(h), dim=1)
        return h, z

    @torch.no_grad()
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class BinaryClassifier(nn.Module):
    """End-to-end binary classifier initialized from a backbone state_dict."""

    def __init__(self, backbone_state_dict: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.backbone = ResNet34Backbone(pretrained=False)
        if backbone_state_dict is not None:
            self.backbone.load_state_dict(copy.deepcopy(backbone_state_dict))
        self.fc = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.fc(h)
