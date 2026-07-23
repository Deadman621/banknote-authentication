# src/application/inference.py

from __future__ import annotations

from pathlib import Path

from typing import Union

from PIL import Image
from torch import nn

from src.application.experiment import ExperimentContext
from src.application.factories import ModuleFactory
from src.augmentation.transforms import build_eval_transforms
from src.inference.loader import prepare_model
from src.inference.predictor import Predictor
from src.inference.state import PredictionResult
from src.models.registry import ModelRegistry

ImageInput = Union[Path, Image.Image]

class InferenceApplication:
    """
    Application layer responsible for model inference.

    Handles:
    - model construction
    - checkpoint loading
    - preprocessing
    - prediction
    """

    def prepare_model(self, context: ExperimentContext, checkpoint: Path) -> nn.Module:
        config = context.config

        model = ModelRegistry.get_model(
            model_name=config.model.name,
            num_classes=config.dataset.num_classes,
            pretrained=False,
            **config.model.params,
        )

        return prepare_model(
            checkpoint_path=checkpoint,
            model=model,
            device=context.device,
        )

    def predict(self, context: ExperimentContext, model: nn.Module, image: ImageInput) -> PredictionResult:
        if isinstance(image, Path):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        transform = build_eval_transforms(
            context.config.dataset.image_size,
        )

        tensor = transform(image).unsqueeze(0)

        predictor = Predictor(model)

        return predictor.predict(tensor)

    def predict_with_module(self, context: ExperimentContext, model: nn.Module, image: ImageInput) -> PredictionResult:
        module_predictor = ModuleFactory.inference(
            context.module
        )

        if module_predictor is Predictor:
            return self.predict(
                context=context,
                model=model,
                image=image,
            )

        predictor = module_predictor(
            model=model,
        )

        return predictor.predict_path(
            image_path=image,
            transform=build_eval_transforms(
                context.config.dataset.image_size,
            ),
        )