# src/models/grade_classifier.py
"""
LGG vs HGG Grade Classifier
────────────────────────────
Binary classifier that predicts tumor grade from a single T1CE slice:
  0 → LGG (Low Grade Glioma)  — slow growing, less urgent
  1 → HGG (High Grade Glioma) — aggressive, needs immediate treatment

Architecture: ResNet-18 pretrained on ImageNet, fine-tuned for binary
classification. T1CE grayscale input is replicated to 3 channels to
match ResNet's expected input.
"""

import torch
import torch.nn as nn
from torchvision import models


class GradeClassifier(nn.Module):
    """
    ResNet-18 based LGG/HGG binary classifier.

    Input : (B, 1, 240, 240) — T1CE grayscale slice
    Output: (B, 2)           — logits for [LGG, HGG]
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = (models.ResNet18_Weights.IMAGENET1K_V1
                   if pretrained else None)
        backbone = models.resnet18(weights=weights)

        # Adapt first conv: 3-channel ImageNet → 1-channel grayscale
        # Average the 3 pretrained input kernels into 1
        orig_weight = backbone.conv1.weight.data  # (64, 3, 7, 7)
        new_weight  = orig_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = new_weight

        # Replace final FC: 512 → 2 (LGG / HGG)
        in_features       = backbone.fc.in_features
        backbone.fc       = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 2)
        )

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
