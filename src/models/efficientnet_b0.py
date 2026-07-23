from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, **kwargs) -> None:
        super().__init__()

        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = models.efficientnet_b0(weights=weights)

        classifier = cast(nn.Sequential, self.model.classifier)
        linear = cast(nn.Linear, classifier[1])

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(linear.in_features, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)