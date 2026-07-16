from typing import Dict, Type

import torch.nn as nn

from .resnet50 import ResNet50
from .efficientnet_b0 import EfficientNetB0
from .swin_transformer import SwinTransformer


class ModelRegistry:
    """
    Central registry for all supported models.
    """

    _MODELS: Dict[str, Type[nn.Module]] = {
        "resnet50": ResNet50,
        "efficientnet_b0": EfficientNetB0,
        "swin_transformer": SwinTransformer,
    }

    @classmethod
    def get_model(
        cls,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        **kwargs,
    ) -> nn.Module:
        """
        Returns an initialized model based on its name.
        """

        model_name = model_name.lower()

        if model_name not in cls._MODELS:
            available = ", ".join(cls._MODELS.keys())
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Available models: {available}"
            )

        model_class = cls._MODELS[model_name]

        return model_class(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs,
        )

    @classmethod
    def list_models(cls):
        """
        Returns all available models.
        """
        return list(cls._MODELS.keys())
