"""
PyTorch hook utilities for Grad-CAM.
"""

from __future__ import annotations

from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle


class ActivationHook:
    """
    Captures forward activations and corresponding gradients from a target layer.
    """

    def __init__(self, layer: Module) -> None:
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None
        self._handle: RemovableHandle = layer.register_forward_hook(self._forward_hook)

    @property
    def activations(self) -> Tensor:
        """
        Return captured activations.

        Raises
        ------
        RuntimeError
            If no forward pass has been executed.
        """
        if self._activations is None:
            raise RuntimeError(
                "No activations have been captured.",
            )

        return self._activations

    @property
    def gradients(self) -> Tensor:
        """
        Return captured gradients.

        Raises
        ------
        RuntimeError
            If no backward pass has been executed.
        """
        if self._gradients is None:
            raise RuntimeError("No gradients have been captured.")

        return self._gradients

    def remove(self) -> None:
        """
        Remove the registered hook.
        """
        self._handle.remove()

    def _forward_hook(self, module: Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        del module
        del inputs

        self._activations = output.detach()

        output.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient: Tensor) -> Tensor:
        self._gradients = gradient.detach()

        return gradient