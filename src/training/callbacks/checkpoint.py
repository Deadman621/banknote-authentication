# src/training/callbacks/checkpoint.py

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from torch import nn
from torch.optim import Optimizer

from src.core.config import CheckpointConfig
from src.checkpoint.state import CheckpointState
from src.training.callbacks.base import Callback
from src.checkpoint.io import (
    SchedulerProtocol,
    save_checkpoint,
)

if TYPE_CHECKING:
    from src.training.state import TrainState

class CheckpointCallback(Callback):
    """
    Saves training checkpoints.

    - last.pt is written after every epoch.
    - best.pt is updated whenever the monitored metric improves.
    """

    def __init__(self, config: CheckpointConfig, directory: Path, model: nn.Module, optimizer: Optimizer, scheduler: SchedulerProtocol | None = None) -> None:
        self.config = config
        self.directory = directory
        self.best_path = directory / "best.pt"
        self.last_path = directory / "last.pt"

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.best_metric: float | None = None

    def on_epoch_end(self, state: TrainState) -> None:
        """
        Save checkpoints after each epoch.

        - last.pt is always updated so training can be resumed.
        - best.pt is updated only when the monitored metric improves.
        """

        metric = self._get_metric(state)

        improved = self._is_improvement(metric)
        if improved:
            self.best_metric = metric

        # Always save the latest checkpoint
        save_checkpoint(
            path=self.last_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=state.epoch,
            global_step=state.global_step,
            best_metric=self.best_metric,
        )

        # Save best checkpoint only on improvement
        if improved:
            save_checkpoint(
                path=self.best_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=state.epoch,
                global_step=state.global_step,
                best_metric=self.best_metric,
            )

    def restore_checkpoint(self, checkpoint: CheckpointState) -> None:
        self.best_metric = checkpoint.best_metric

    def _get_metric(self, state: TrainState) -> float:
        value = getattr(state, self.config.monitor, None)

        if value is None:
            raise ValueError(
                f"Unknown checkpoint metric "
                f"'{self.config.monitor}'."
            )

        return float(value)

    def _is_improvement(self, value: float) -> bool:
        if self.best_metric is None:
            return True

        if self.config.mode == "min":
            return value < self.best_metric

        if self.config.mode == "max":
            return value > self.best_metric

        raise ValueError(
            f"Unsupported checkpoint mode "
            f"'{self.config.mode}'."
        )