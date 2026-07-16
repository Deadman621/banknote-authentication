import torch
import torch.nn as nn
import torchvision.models as models

class SwinTransformer(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, **kwargs) -> None:
        super().__init__()
        # Swin Transformer Tiny weights load karna safely
        weights = models.Swin_V2_T_Weights.DEFAULT if pretrained else None
        self.model = models.swin_v2_t(weights=weights)
        
        # Head (Classifier) layer ko update karna
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)
