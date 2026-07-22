# src/training/callbacks/logging.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.training.callbacks.base import Callback
from tqdm import tqdm


if TYPE_CHECKING:
    from src.training.state import TrainState


class LoggingCallback(Callback):
    """
    Training logger callback.

    Writes training lifecycle information
    to the configured logger.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def on_train_begin(self, state: TrainState) -> None:
        self.logger.info("Training started.")

    def on_epoch_end(self, state: TrainState) -> None:
        self.logger.info(
            "Epoch %d completed | train_loss=%.4f | val_loss=%.4f | "
            "train_acc=%.4f | val_acc=%.4f",
            state.epoch,
            state.train_loss,
            state.validation_loss,
            state.train_accuracy,
            state.validation_accuracy,
        )

    def on_train_end(self, state: TrainState) -> None:
        self.logger.info("Training finished.")