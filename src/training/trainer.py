# src/training/trainer.py

from __future__ import annotations

from collections.abc import Iterable

from typing import Optional

import torch

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.checkpoint.state import CheckpointState
from src.core.config import ExperimentConfig
from src.training.state import TrainState
from src.training.amp import AMPContext
from src.training.callbacks.base import Callback
from src.training.engine import TrainingEngine
from src.utils.protocols import SchedulerProtocol

class Trainer:
    """
    High-level training coordinator.

    Does not implement training logic.
    Delegates to TrainingEngine.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: SchedulerProtocol | None = None,
        callbacks: Iterable[Callback] = (),
    ) -> None:
        self.config = config

        self.engine = TrainingEngine(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            amp=AMPContext(
                enabled=config.trainer.mixed_precision,
                device=device,
            ),
            gradient_clip=config.trainer.gradient_clip,
            callbacks=callbacks,
        ) 

    def restore_checkpoint(self, checkpoint: CheckpointState) -> None:
        for callback in self.engine.callbacks:
            callback.restore_checkpoint(checkpoint)

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader, initial_state: Optional[TrainState] = None) -> TrainState:
        """
        Start training.
        """

        return self.engine.fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=self.config.trainer.epochs,
            initial_state=initial_state,
        )