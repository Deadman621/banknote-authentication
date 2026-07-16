from __future__ import annotations

import torch

from src.training.utils import move_batch_to_device


def test_move_batch_to_device_cpu() -> None:
    """Images and labels are moved to the requested device."""

    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])

    device = torch.device("cpu")

    moved_images, moved_labels = move_batch_to_device(
        images,
        labels,
        device,
    )

    assert moved_images.device == device
    assert moved_labels.device == device


def test_move_batch_to_device_preserves_values() -> None:
    """Moving a batch preserves tensor values."""

    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])

    moved_images, moved_labels = move_batch_to_device(
        images,
        labels,
        torch.device("cpu"),
    )

    assert torch.equal(images, moved_images)
    assert torch.equal(labels, moved_labels)