from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.explainability.state import GradCAMResult


def test_gradcam_result_stores_values() -> None:
    heatmap = np.random.rand(224, 224).astype(np.float32)
    probabilities = np.array(
        [0.02, 0.03, 0.95],
        dtype=np.float32,
    )

    result = GradCAMResult(
        heatmap=heatmap,
        predicted_class=2,
        target_class=2,
        class_probabilities=probabilities,
        image_size=(224, 224),
    )

    assert result.predicted_class == 2
    assert result.target_class == 2
    assert result.image_size == (224, 224)

    np.testing.assert_array_equal(result.heatmap, heatmap)
    np.testing.assert_array_equal(
        result.class_probabilities,
        probabilities,
    )


def test_gradcam_result_is_frozen() -> None:
    result = GradCAMResult(
        heatmap=np.zeros((8, 8), dtype=np.float32),
        predicted_class=0,
        target_class=0,
        class_probabilities=np.array(
            [1.0],
            dtype=np.float32,
        ),
        image_size=(8, 8),
    )

    with pytest.raises(FrozenInstanceError):
        result.predicted_class = 1  # type: ignore[misc]


def test_gradcam_result_preserves_heatmap_dtype() -> None:
    result = GradCAMResult(
        heatmap=np.zeros((32, 32), dtype=np.float32),
        predicted_class=0,
        target_class=0,
        class_probabilities=np.array(
            [1.0],
            dtype=np.float32,
        ),
        image_size=(32, 32),
    )

    assert result.heatmap.dtype == np.float32


def test_gradcam_result_preserves_probability_dtype() -> None:
    result = GradCAMResult(
        heatmap=np.zeros((32, 32), dtype=np.float32),
        predicted_class=0,
        target_class=0,
        class_probabilities=np.array(
            [0.3, 0.7],
            dtype=np.float32,
        ),
        image_size=(32, 32),
    )

    assert result.class_probabilities.dtype == np.float32


def test_gradcam_result_preserves_heatmap_shape() -> None:
    heatmap = np.zeros((128, 256), dtype=np.float32)

    result = GradCAMResult(
        heatmap=heatmap,
        predicted_class=1,
        target_class=1,
        class_probabilities=np.array(
            [0.1, 0.9],
            dtype=np.float32,
        ),
        image_size=(128, 256),
    )

    assert result.heatmap.shape == (128, 256)


def test_gradcam_result_preserves_probability_shape() -> None:
    probabilities = np.array(
        [0.1, 0.2, 0.3, 0.4],
        dtype=np.float32,
    )

    result = GradCAMResult(
        heatmap=np.zeros((8, 8), dtype=np.float32),
        predicted_class=3,
        target_class=3,
        class_probabilities=probabilities,
        image_size=(8, 8),
    )

    assert result.class_probabilities.shape == (4,)


def test_gradcam_result_stores_image_size() -> None:
    result = GradCAMResult(
        heatmap=np.zeros((480, 640), dtype=np.float32),
        predicted_class=1,
        target_class=1,
        class_probabilities=np.array(
            [0.2, 0.8],
            dtype=np.float32,
        ),
        image_size=(480, 640),
    )

    assert result.image_size == (480, 640)


def test_gradcam_result_supports_custom_target_class() -> None:
    result = GradCAMResult(
        heatmap=np.ones((16, 16), dtype=np.float32),
        predicted_class=5,
        target_class=2,
        class_probabilities=np.array(
            [0.01, 0.02, 0.92, 0.05],
            dtype=np.float32,
        ),
        image_size=(16, 16),
    )

    assert result.predicted_class == 5
    assert result.target_class == 2


def test_predicted_class_matches_argmax_probability() -> None:
    probabilities = np.array(
        [0.05, 0.10, 0.80, 0.05],
        dtype=np.float32,
    )

    result = GradCAMResult(
        heatmap=np.zeros((8, 8), dtype=np.float32),
        predicted_class=2,
        target_class=2,
        class_probabilities=probabilities,
        image_size=(8, 8),
    )

    assert result.predicted_class == int(
        np.argmax(result.class_probabilities)
    )