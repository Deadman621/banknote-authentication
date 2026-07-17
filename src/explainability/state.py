"""
Immutable state objects for Grad-CAM explainability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class GradCAMResult:
    heatmap: NDArray[np.float32]
    predicted_class: int
    target_class: int
    class_probabilities: NDArray[np.float32]
    image_size: tuple[int, int]