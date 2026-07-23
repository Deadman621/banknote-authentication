from __future__ import annotations

from pathlib import Path

from typing import Union

from src.augmentation.transforms import build_eval_transforms
from src.explainability.gradcam import GradCAM
from src.explainability.target import resolve_target_layer
from src.explainability.visualization import overlay_heatmap

from PIL import Image

import numpy as np

ImageInput = Union[Path, Image.Image]

class GradCAMApplication:
    """
    Application layer for Grad-CAM generation.
    """

    def generate(
        self,
        context,
        model,
        image: ImageInput,
        target_layer: str | None = None,
        target_class: int | None = None,
    ):

        model = model.to(context.device)
        model.eval()

        if isinstance(image, Path):
            image = Image.open(image)

        image = image.convert("RGB")


        image_size = context.config.dataset.image_size


        original = np.asarray(
            image.resize(
                (
                    image_size,
                    image_size,
                )
            )
        )


        transform = build_eval_transforms(
            image_size
        )


        tensor = transform(image)

        target = resolve_target_layer(
            model,
            target_layer,
        )


        gradcam = GradCAM(
            model=model,
            target_layer=target,
        )


        result = gradcam.generate(
            tensor,
            target_class=target_class,
        )


        overlay = overlay_heatmap(
            original,
            result.heatmap,
        )


        return result, overlay