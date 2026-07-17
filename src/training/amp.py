# src/training/amp.py

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch
from torch import Tensor
from torch.optim import Optimizer


class AMPContext:
    """
    Mixed precision helper.

    Provides a unified interface for:
    - autocast
    - gradient scaling
    """

    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = enabled and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            enabled=self.enabled,
        )

        self.device = device

    def autocast(self) -> ContextManager[object]:
        """
        Return autocast context manager.
        """

        if not self.enabled:
            return nullcontext()

        return torch.amp.autocast("cuda")

    def backward(self, loss: Tensor, optimizer: Optimizer) -> None:
        """
        Perform backward pass.
        """

        if self.enabled:
            self.scaler.scale(loss).backward()
            return

        loss.backward()

    def step(self, optimizer: Optimizer) -> None:
        """
        Optimizer step with AMP handling.
        """

        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
            return

        optimizer.step()

    def unscale_(self, optimizer: Optimizer) -> None:
        """
        Unscale gradients before clipping when AMP is enabled.
        """

        if self.enabled:
            self.scaler.unscale_(optimizer)

    def zero_grad(self, optimizer: Optimizer) -> None:
        """
        Clear gradients.
        """

        optimizer.zero_grad()