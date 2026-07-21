"""Tests for scripts/_common.py pipeline bridging functions."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from PIL import Image

from scripts._common import (
    ImageSampleDataset,
    build_evaluation_loader,
    build_module_dataset,
    build_training_loaders,
    load_image_tensor,
    predict_image,
    serialize_evaluation_result,
)
from src.datasets.dataset import CurrencyDataset
from src.evaluation.state import EvaluationMetrics, EvaluationResult
from src.inference.state import PredictionResult
import numpy as np


def test_image_sample_dataset_is_currency_dataset(tmp_path: Path) -> None:
    img_path = tmp_path / "test.png"
    Image.new("RGB", (30, 30)).save(img_path)

    ds = ImageSampleDataset(root=tmp_path, samples=[(img_path, 0)])
    assert isinstance(ds, CurrencyDataset)
    assert len(ds) == 1


def test_serialize_evaluation_result_uses_src_serialization() -> None:
    predictions = torch.tensor([0, 1])
    targets = torch.tensor([0, 1])
    probabilities = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    metrics = EvaluationMetrics(
        accuracy=1.0, precision=1.0, recall=1.0, f1=1.0, roc_auc=1.0
    )
    result = EvaluationResult(
        loss=0.05,
        metrics=metrics,
        predictions=predictions,
        targets=targets,
        probabilities=probabilities,
        confusion_matrix=np.array([[1, 0], [0, 1]]),
    )

    serialized = serialize_evaluation_result(result, ["class_a", "class_b"])
    assert serialized["loss"] == 0.05
    assert serialized["metrics"]["accuracy"] == 1.0
    assert serialized["class_names"] == ["class_a", "class_b"]


def test_predict_image_integration(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64)).save(img_path)

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 2))
    model.eval()

    res, tensor = predict_image(model, img_path, image_size=32)
    assert isinstance(res, PredictionResult)
    assert tensor.shape == (1, 3, 32, 32)
