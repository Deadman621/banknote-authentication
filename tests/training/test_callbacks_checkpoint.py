from __future__ import annotations

import torch
import torch.nn as nn

from src.core.config import CheckpointConfig
from src.training.callbacks.checkpoint import (
    CheckpointCallback,
)
from src.training.state import TrainState


def test_checkpoint_callback_saves_best(
    tmp_path,
) -> None:
    model = nn.Linear(
        10,
        2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    callback = CheckpointCallback(
        config=CheckpointConfig(
            monitor="validation_loss",
            mode="min",
            save_best_only=True,
        ),
        path=tmp_path / "best.pt",
        model=model,
        optimizer=optimizer,
    )

    state = TrainState(
        epoch=1,
        global_step=10,
        validation_loss=0.5,
    )

    callback.on_epoch_end(state)

    assert (
        tmp_path / "best.pt"
    ).exists()


def test_checkpoint_callback_skips_worse_metric(
    tmp_path,
) -> None:
    model = nn.Linear(
        10,
        2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    callback = CheckpointCallback(
        config=CheckpointConfig(
            monitor="validation_loss",
            mode="min",
            save_best_only=True,
        ),
        path=tmp_path / "best.pt",
        model=model,
        optimizer=optimizer,
    )

    callback.on_epoch_end(
        TrainState(
            epoch=1,
            validation_loss=0.5,
        )
    )

    callback.on_epoch_end(
        TrainState(
            epoch=2,
            validation_loss=0.8,
        )
    )

    assert (
        tmp_path / "best.pt"
    ).exists()