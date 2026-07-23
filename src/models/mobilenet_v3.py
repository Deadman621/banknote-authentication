from __future__ import annotations

from typing import Optional, cast

from torch import Tensor, nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)


class MobileNetClassifier(nn.Module):
    """MobileNetV3-Large classifier for currency note denomination recognition."""

    def __init__(self, num_classes: int, *, pretrained: bool = True) -> None:
        super().__init__()

        weights: Optional[MobileNet_V3_Large_Weights] = (
            MobileNet_V3_Large_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = mobilenet_v3_large(weights=weights)
        last_layer = cast(nn.Linear, self.model.classifier[-1])

        self.model.classifier[-1] = nn.Linear(
            last_layer.in_features,
            num_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)