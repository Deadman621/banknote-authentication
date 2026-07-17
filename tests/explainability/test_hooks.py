from __future__ import annotations

import pytest
import torch
from torch import nn

from src.explainability.hooks import ActivationHook


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            3,
            4,
            kernel_size=3,
            padding=1,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(
            4,
            2,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.fc(x)


def test_activation_hook_captures_activations() -> None:
    model = ToyModel()

    hook = ActivationHook(model.conv)

    images = torch.randn(
        2,
        3,
        32,
        32,
    )

    model(images)

    assert hook.activations.shape == (
        2,
        4,
        32,
        32,
    )


def test_activation_hook_captures_gradients() -> None:
    model = ToyModel()

    hook = ActivationHook(model.conv)

    images = torch.randn(
        2,
        3,
        32,
        32,
    )

    logits = model(images)

    logits.sum().backward()

    assert hook.gradients.shape == (
        2,
        4,
        32,
        32,
    )


def test_activation_hook_raises_before_forward() -> None:
    model = ToyModel()

    hook = ActivationHook(model.conv)

    with pytest.raises(RuntimeError):
        _ = hook.activations


def test_activation_hook_raises_before_backward() -> None:
    model = ToyModel()

    hook = ActivationHook(model.conv)

    images = torch.randn(
        1,
        3,
        32,
        32,
    )

    model(images)

    with pytest.raises(RuntimeError):
        _ = hook.gradients


def test_activation_hook_updates_after_multiple_forward_passes() -> None:
    model = ToyModel()

    hook = ActivationHook(model.conv)

    first = torch.randn(
        1,
        3,
        32,
        32,
    )

    second = torch.randn(
        2,
        3,
        32,
        32,
    )

    model(first)

    assert hook.activations.shape[0] == 1

    model(second)

    assert hook.activations.shape[0] == 2


def test_activation_hook_updates_after_multiple_backward_passes() -> None:
    model = ToyModel()

    hook = ActivationHook(model.conv)

    first = torch.randn(
        1,
        3,
        32,
        32,
    )

    second = torch.randn(
        2,
        3,
        32,
        32,
    )

    model(first).sum().backward()

    assert hook.gradients.shape[0] == 1

    model.zero_grad(set_to_none=True)

    model(second).sum().backward()

    assert hook.gradients.shape[0] == 2


def test_activation_hook_remove() -> None:
    model = ToyModel()

    hook = ActivationHook(model.conv)

    hook.remove()


def test_activation_and_gradient_have_same_shape() -> None:
    model = ToyModel()

    hook = ActivationHook(model.conv)

    images = torch.randn(
        2,
        3,
        32,
        32,
    )

    logits = model(images)

    logits.sum().backward()

    assert hook.activations.shape == hook.gradients.shape