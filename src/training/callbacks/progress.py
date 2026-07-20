# src/training/callbacks/progress.py

from __future__ import annotations

from tqdm import tqdm
from typing import TYPE_CHECKING
from src.training.callbacks.base import Callback

if TYPE_CHECKING:
    from src.training.state import TrainState

class ProgressCallback(Callback):
    """
    Progress bar callback.

    Handles visual training progress only.
    """

    def __init__(self) -> None:
        self.progress_bar: tqdm | None = None

    def on_epoch_begin(self, state: TrainState) -> None:
        """
        Reset epoch progress.
        """

        self.progress_bar = tqdm(
            total=state.total_batches,
            desc=f"Epoch {state.epoch}",
            leave=True,
        )

    def on_batch_end(self, state: TrainState) -> None:
        """
        Update progress after batch.
        """

        if self.progress_bar is None:
            return

        self.progress_bar.update(1)

        self.progress_bar.set_postfix(
            {
                "loss": f"{state.train_loss:.4f}",
                "acc": f"{state.train_accuracy:.4f}",
            }
        )

    def on_epoch_end(self, state: TrainState) -> None:
        if self.progress_bar is None:
            return

        self.progress_bar.close()
        print(end='\n')
        self.progress_bar = None