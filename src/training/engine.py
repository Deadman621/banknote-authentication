# src/training/engine.py

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Protocol

from src.training.amp import AMPContext
from src.training.callbacks.base import Callback
from src.training.metrics import accuracy
from src.training.state import TrainState
from src.utils.protocols import SchedulerProtocol
from src.training.utils import move_batch_to_device

from torch.utils.data import DataLoader

class TrainingEngine:
    """
    Core training execution engine.

    Responsible only for:
    - forward pass
    - backward pass
    - optimizer updates
    - metric calculation
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        amp: AMPContext,
        scheduler: SchedulerProtocol | None = None,
        gradient_clip: float | None = None,
        callbacks: Iterable[Callback] = (),
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.amp = amp
        self.gradient_clip = gradient_clip

        self.callbacks = list(callbacks)

        self.state = TrainState()

        self.model.to(self.device)

    def train_step(self, images: Tensor, labels: Tensor) -> tuple[float, float]:
        """
        Execute a single training batch.

        Returns
        -------
        tuple[float, float]
            loss and accuracy
        """

        self.model.train()

        images, labels = move_batch_to_device(images, labels, self.device)
        self.amp.zero_grad(self.optimizer)

        with self.amp.autocast():
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

        self.amp.backward( loss, self.optimizer)

        if self.gradient_clip is not None:
            self.amp.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip,
            )

        self.amp.step(self.optimizer)
        batch_accuracy = accuracy(logits.detach(), labels)

        return (float(loss.item()), batch_accuracy)
    
    def train_epoch(self, loader: DataLoader) -> None:
        """
        Train for one epoch.
        """

        total_loss = 0.0
        total_accuracy = 0.0
        batches = 0

        for callback in self.callbacks:
            callback.on_epoch_begin(self.state)

        for images, labels in loader:
            for callback in self.callbacks:
                callback.on_batch_begin(self.state)

            loss, batch_accuracy = self.train_step(images, labels)

            total_loss += loss
            total_accuracy += batch_accuracy
            batches += 1

            self.state.global_step += 1
            self.state.train_loss = (total_loss / batches)
            self.state.train_accuracy = (total_accuracy / batches)

            for callback in self.callbacks:
                callback.on_batch_end(self.state)

        for callback in self.callbacks:
            callback.on_epoch_end(self.state)

    def validate_epoch(self, loader: DataLoader) -> None:
        """
        Validate for one epoch.
        """

        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        batches = 0

        with torch.no_grad():

            for images, labels in loader:
                images, labels = move_batch_to_device(images, labels, self.device)
                with self.amp.autocast():
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels,)
                
                batch_accuracy = accuracy(logits, labels)
                total_loss += float(loss.item())
                total_accuracy += (batch_accuracy)

                batches += 1

        self.state.validation_loss = (total_loss / batches)
        self.state.validation_accuracy = (total_accuracy / batches)

        for callback in self.callbacks:
            callback.on_validation_end(self.state)


    def fit(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int) -> TrainState:
        """
        Execute full training loop.
        """

        for callback in self.callbacks:
            callback.on_train_begin(
                self.state
            )

        for epoch in range(epochs):
            self.state.epoch = epoch + 1
            self.train_epoch(train_loader)
            self.validate_epoch(validation_loader)

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(self.state.validation_loss)
                else:
                    self.scheduler.step()

            if self.state.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end(self.state)

        return self.state