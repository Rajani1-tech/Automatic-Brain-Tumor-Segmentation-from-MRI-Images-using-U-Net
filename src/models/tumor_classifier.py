# src/models/tumor_classifier.py
"""
TumorClassifier
───────────────
ResNet-18 fine-tuned for 4-class tumor-profile classification.

Classes (derived from mask pixel values — no separate folder needed):
    0 → No Tumor
    1 → Edema Only
    2 → Core Present
    3 → Full Tumor

Input : (B, 1, H, W) — single-channel grayscale MRI slice
Output: (B, 4)        — raw logits (apply softmax externally)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from configs.config import Config


class TumorClassifier(nn.Module):
    def __init__(self,
                 num_classes: int = Config.NUM_TUMOR_CLASSES,
                 pretrained: bool = True):
        super().__init__()

        weights  = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # MRI is single-channel; adapt first conv (ImageNet expects 3-ch)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace classifier head
        in_features  = backbone.fc.in_features
        backbone.fc  = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)