# src/training/callbacks/base.py

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.training.state import TrainState
    from src.checkpoint.state import CheckpointState

class Callback:
    """
    Base class for training callbacks.

    All methods are intentionally no-op.
    Subclasses override only required hooks.
    """

    def on_train_begin(self, state: TrainState) -> None:
        pass

    def on_train_end(self, state: TrainState) -> None:
        pass

    def on_epoch_begin(self, state: TrainState) -> None:
        pass

    def on_epoch_end(self, state: TrainState) -> None:
        pass

    def on_batch_begin(self, state: TrainState) -> None:
        pass

    def on_batch_end(self, state: TrainState) -> None:
        pass

    def on_validation_end(self, state: TrainState) -> None:
        pass

    def restore_checkpoint(self, checkpoint: CheckpointState) -> None:
        pass