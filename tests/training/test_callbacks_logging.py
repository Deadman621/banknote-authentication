from __future__ import annotations

import logging
from src.training.state import TrainState
from src.training.callbacks.logging import (
    LoggingCallback,
)

def test_logging_callback() -> None:
    logger = logging.getLogger("test_logger")

    callback = LoggingCallback(logger)

    state = TrainState(
        epoch=1,
        train_loss=0.1,
        validation_loss=0.2,
        train_accuracy=0.9,
        validation_accuracy=0.8,
    )

    callback.on_train_begin(state)

    callback.on_epoch_end(state)

    callback.on_train_end(state)

    assert True