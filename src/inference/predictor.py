from __future__ import annotations

from torch import Tensor
import torch
from torch import nn

from src.inference.state import BatchPredictionResult, PredictionResult


class Predictor:
    """Runs inference using a trained model."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def predict(self, image: Tensor) -> PredictionResult:
        """Predict a single image."""

        with torch.inference_mode():
            logits = self._model(image)

            probabilities = torch.softmax(
                logits,
                dim=1,
            ).squeeze(0)

        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())

        return PredictionResult(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities.cpu().numpy(),
        )

    def predict_batch(self, images: Tensor) -> BatchPredictionResult:
        """Predict a batch of images."""

        with torch.inference_mode():
            logits = self._model(images)
            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        predictions: list[PredictionResult] = []

        for probability in probabilities:
            predicted_class = int(torch.argmax(probability).item())
            confidence = float(probability[predicted_class].item())

            predictions.append(
                PredictionResult(
                    predicted_class=predicted_class,
                    confidence=confidence,
                    probabilities=probability.cpu().numpy(),
                )
            )

        return BatchPredictionResult(predictions=tuple(predictions))