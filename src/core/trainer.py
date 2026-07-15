"""
Generic PyTorch training loop.
"""

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

class Trainer:
    """Generic model trainer."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Neural network model.
            criterion: Loss function.
            optimizer: Optimizer.
            device: Training device.
            scheduler: Optional learning rate scheduler.
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def train_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, Tensor]],
    ) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader.

        Returns:
            Average training loss.
        """
        self.model.train()

        running_loss = 0.0

        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return running_loss / len(dataloader)

    def validate_epoch(self, dataloader: DataLoader[tuple[Tensor, Tensor]]) -> float:
        """
        Validate for one epoch.

        Args:
            dataloader: Validation dataloader.

        Returns:
            Average validation loss.
        """
        self.model.eval()

        running_loss = 0.0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

        return running_loss / len(dataloader)

    def fit(
        self,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        val_loader: DataLoader[tuple[Tensor, Tensor]],
        epochs: int,
    ) -> None:
        """
        Train the model.

        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            epochs: Number of epochs.
        """

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)

            val_loss = self.validate_epoch(val_loader)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )