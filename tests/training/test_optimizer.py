from __future__ import annotations

import pytest
import torch.nn as nn

from torch.optim import Adam, AdamW, SGD

from src.core.config import OptimizerConfig
from src.training.optimizer import build_optimizer


def create_model() -> nn.Module:
    return nn.Linear(
        10,
        2,
    )


def test_adam_optimizer() -> None:
    model = create_model()

    config = OptimizerConfig(
        name="adam",
        lr=0.001,
        weight_decay=0.01,
    )

    optimizer = build_optimizer(
        model,
        config,
    )

    assert isinstance(
        optimizer,
        Adam,
    )


def test_adamw_optimizer() -> None:
    model = create_model()

    config = OptimizerConfig(
        name="adamw",
        lr=0.001,
        weight_decay=0.01,
    )

    optimizer = build_optimizer(
        model,
        config,
    )

    assert isinstance(
        optimizer,
        AdamW,
    )


def test_sgd_optimizer() -> None:
    model = create_model()

    config = OptimizerConfig(
        name="sgd",
        lr=0.01,
        weight_decay=0.0,
    )

    optimizer = build_optimizer(
        model,
        config,
    )

    assert isinstance(
        optimizer,
        SGD,
    )


def test_optimizer_parameters() -> None:
    model = create_model()

    config = OptimizerConfig(
        name="adam",
        lr=0.005,
        weight_decay=0.1,
    )

    optimizer = build_optimizer(
        model,
        config,
    )

    assert optimizer.defaults["lr"] == 0.005
    assert optimizer.defaults["weight_decay"] == 0.1


def test_invalid_optimizer() -> None:
    model = create_model()

    config = OptimizerConfig(
        name="invalid",
        lr=0.001,
        weight_decay=0.0,
    )

    with pytest.raises(ValueError):
        build_optimizer(
            model,
            config,
        )