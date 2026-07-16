from __future__ import annotations

import pytest
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

from src.core.config import SchedulerConfig
from src.training.scheduler import build_scheduler


def create_optimizer() -> Adam:
    model = nn.Linear(
        10,
        2,
    )

    return Adam(
        model.parameters(),
        lr=0.001,
    )


def test_step_lr_scheduler() -> None:
    optimizer = create_optimizer()

    config = SchedulerConfig(
        name="step_lr",
        params={
            "step_size": 5,
            "gamma": 0.1,
        },
    )

    scheduler = build_scheduler(
        optimizer,
        config,
    )

    assert isinstance(
        scheduler,
        StepLR,
    )


def test_cosine_scheduler() -> None:
    optimizer = create_optimizer()

    config = SchedulerConfig(
        name="cosine",
        params={
            "T_max": 10,
        },
    )

    scheduler = build_scheduler(
        optimizer,
        config,
    )

    assert isinstance(
        scheduler,
        CosineAnnealingLR,
    )


def test_reduce_on_plateau_scheduler() -> None:
    optimizer = create_optimizer()

    config = SchedulerConfig(
        name="reduce_on_plateau",
        params={
            "mode": "min",
            "factor": 0.1,
            "patience": 3,
        },
    )

    scheduler = build_scheduler(
        optimizer,
        config,
    )

    assert isinstance(
        scheduler,
        ReduceLROnPlateau,
    )


def test_none_scheduler() -> None:
    optimizer = create_optimizer()

    config = SchedulerConfig(
        name="none",
        params={},
    )

    scheduler = build_scheduler(
        optimizer,
        config,
    )

    assert scheduler is None


def test_invalid_scheduler() -> None:
    optimizer = create_optimizer()

    config = SchedulerConfig(
        name="invalid",
        params={},
    )

    with pytest.raises(ValueError):
        build_scheduler(
            optimizer,
            config,
        )