# tests/test_device.py

from __future__ import annotations

import pytest
import torch

from src.training.device import get_device


def test_get_cpu_device() -> None:
    device = get_device("cpu")

    assert device.type == "cpu"


def test_get_cpu_device_case_insensitive() -> None:
    device = get_device("CPU")

    assert device.type == "cpu"


def test_invalid_device_name() -> None:
    with pytest.raises(ValueError):
        get_device("invalid")


def test_cuda_device_when_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA available")

    with pytest.raises(RuntimeError):
        get_device("cuda")


def test_cuda_device_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = get_device("cuda")

    assert device.type == "cuda"


def test_mps_device_behavior() -> None:
    if not hasattr(torch.backends, "mps"):
        pytest.skip("MPS backend unavailable")

    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError):
            get_device("mps")

        return

    device = get_device("mps")

    assert device.type == "mps"