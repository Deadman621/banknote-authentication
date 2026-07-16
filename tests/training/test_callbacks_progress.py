from __future__ import annotations

from src.training.callbacks.progress import (
    ProgressCallback,
)

from src.training.state import TrainState


def test_progress_callback_lifecycle() -> None:
    callback = ProgressCallback()

    state = TrainState(
        epoch=1,
    )

    callback.on_epoch_begin(state)

    assert callback.progress_bar is not None

    callback.on_batch_end(state)

    callback.on_epoch_end(state)

    assert callback.progress_bar is None