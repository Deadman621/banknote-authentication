from __future__ import annotations

from src.training.callbacks.base import Callback
from src.training.state import TrainState


def test_base_callback_methods_do_nothing() -> None:
    callback = Callback()
    state = TrainState()

    callback.on_train_begin(state)
    callback.on_train_end(state)

    callback.on_epoch_begin(state)
    callback.on_epoch_end(state)

    callback.on_batch_begin(state)
    callback.on_batch_end(state)

    callback.on_validation_end(state)

    assert True