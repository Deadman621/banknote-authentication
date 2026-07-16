# src/training/device.py

from __future__ import annotations

import torch


def get_device(name: str) -> torch.device:
    """
    Resolve the training device from configuration.

    Parameters
    ----------
    name:
        Device name from configuration.
        Supported values:
        - "cpu"
        - "cuda"
        - "mps"

    Returns
    -------
    torch.device
        Resolved PyTorch device.

    Raises
    ------
    ValueError
        If an unsupported device name is provided.

    RuntimeError
        If the requested accelerator is unavailable.
    """

    normalized = name.lower().strip()

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is not available."
            )

        return torch.device("cuda")

    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS device requested but MPS is not available."
            )

        return torch.device("mps")

    raise ValueError(
        f"Unsupported device '{name}'. "
        "Expected one of: cpu, cuda, mps."
    )