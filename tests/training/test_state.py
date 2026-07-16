from __future__ import annotations

from src.training.state import TrainState


def test_default_state() -> None:
    state = TrainState()

    assert state.epoch == 0
    assert state.global_step == 0

    assert state.train_loss == 0.0
    assert state.validation_loss == 0.0

    assert state.should_stop is False


def test_state_is_mutable() -> None:
    state = TrainState()

    state.epoch = 5
    state.global_step = 100

    state.train_loss = 0.25
    state.validation_loss = 0.30

    assert state.epoch == 5
    assert state.global_step == 100

    assert state.train_loss == 0.25
    assert state.validation_loss == 0.30


def test_stop_flag() -> None:
    state = TrainState()

    state.should_stop = True

    assert state.should_stop is True