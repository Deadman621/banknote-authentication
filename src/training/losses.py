# src/training/losses.py

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.config import LossConfig


def build_loss(config: LossConfig, class_weights: torch.Tensor | None = None) -> nn.Module:
    """
    Build loss function from configuration.

    Parameters
    ----------
    config:
        Loss configuration.

    class_weights:
        Optional tensor of weights for each class.

    Returns
    -------
    torch.nn.Module
        Instantiated loss function.
    """

    name = config.name.lower().strip()

    if name == "cross_entropy":

        if class_weights is not None:
            return nn.CrossEntropyLoss(
                weight=class_weights
            )

        return nn.CrossEntropyLoss()

    raise ValueError(
        f"Unsupported loss '{config.name}'."
    )