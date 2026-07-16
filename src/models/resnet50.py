import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)
