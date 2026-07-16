# src/training/callbacks/checkpoint.py

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from torch import nn
from torch.optim import Optimizer

from src.core.config import CheckpointConfig
from src.training.callbacks.base import Callback
from src.training.checkpoint import (
    SchedulerProtocol,
    save_checkpoint,
)

if TYPE_CHECKING:
    from src.training.state import TrainState

class CheckpointCallback(Callback):
    """
    Saves model checkpoints during training.
    """

    def __init__(self, config: CheckpointConfig, path: Path, model: nn.Module, optimizer: Optimizer, scheduler: SchedulerProtocol | None = None) -> None:
        self.config = config
        self.path = path

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.best_metric: float | None = None

    def on_epoch_end(self, state: TrainState) -> None:
        metric = self._get_metric(state)

        if self.config.save_best_only:
            if not self._is_improvement(metric):
                return

        save_checkpoint(
            path=self.path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=state.epoch,
            global_step=state.global_step,
        )

        self.best_metric = metric

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