from __future__ import annotations

from src.core.config import EarlyStoppingConfig
from src.training.callbacks.early_stopping import (
    EarlyStoppingCallback,
)
from src.training.state import TrainState


def test_early_stopping_triggers() -> None:
    callback = EarlyStoppingCallback(
        EarlyStoppingConfig(
            patience=2,
            monitor="validation_loss",
        )
    )

    state = TrainState()

    state.validation_loss = 1.0
    callback.on_epoch_end(state)

    assert state.should_stop is False

    state.validation_loss = 1.1
    callback.on_epoch_end(state)

    assert state.should_stop is False

    state.validation_loss = 1.2
    callback.on_epoch_end(state)

    assert state.should_stop is True


def test_improvement_resets_counter() -> None:
    callback = EarlyStoppingCallback(
        EarlyStoppingConfig(
            patience=2,
            monitor="validation_loss",
        )
    )

    state = TrainState()

    state.validation_loss = 1.0
    callback.on_epoch_end(state)

    state.validation_loss = 1.2
    callback.on_epoch_end(state)

    state.validation_loss = 0.8
    callback.on_epoch_end(state)

    assert state.should_stop is False
    assert callback.wait == 0