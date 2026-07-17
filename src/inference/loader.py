from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.checkpoint.io import load_checkpoint


def prepare_model(checkpoint_path: Path, model: nn.Module, device: torch.device) -> nn.Module:
    """Load checkpoint weights and prepare a model for inference."""

    load_checkpoint(path=checkpoint_path, model=model)

    model.to(device)
    model.eval()

    return model