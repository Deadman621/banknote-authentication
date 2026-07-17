from typing import Dict, Type

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency guard
    timm = None

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

    _TIMM_MODELS: Dict[str, str] = {
        "efficientnet_b3": "efficientnet_b3",
        "mobilenet_v3": "mobilenetv3_large_100",
        "vit": "vit_base_patch16_224",
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
            if model_name in cls._TIMM_MODELS:
                if timm is None:
                    raise ImportError(
                        "timm is required for model "
                        f"'{model_name}'."
                    )

                return timm.create_model(
                    cls._TIMM_MODELS[model_name],
                    pretrained=pretrained,
                    num_classes=num_classes,
                    **kwargs,
                )

            available = ", ".join(
                sorted({*cls._MODELS.keys(), *cls._TIMM_MODELS.keys()})
            )
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
        return list({*cls._MODELS.keys(), *cls._TIMM_MODELS.keys()})
