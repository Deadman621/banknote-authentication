from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Represents the prediction for a single input."""

    predicted_class: int
    confidence: float
    probabilities: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class BatchPredictionResult:
    """Represents predictions for multiple inputs."""

    predictions: tuple[PredictionResult, ...]


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    image_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]