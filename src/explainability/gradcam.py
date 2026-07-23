"""
Grad-CAM implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from src.explainability.hooks import ActivationHook
from src.explainability.state import GradCAMResult

class GradCAM:
    """
    Generates Grad-CAM explanations for image classification models.
    """

    def __init__(self, model: Module, target_layer: Module) -> None:
        self._model = model
        self._target_layer = target_layer

    def _compute_weights(self, gradients: Tensor) -> Tensor:
        """
        Compute Grad-CAM channel weights.
        """
        return gradients.mean(dim=(2, 3), keepdim=True)
    
    def _compute_heatmap(self, activations: Tensor, weights: Tensor) -> Tensor:
        """
        Compute the raw Grad-CAM heatmap.

        Parameters
        ----------
        activations:
            Feature maps captured from the target layer.
        weights:
            Channel-wise Grad-CAM weights.

        Returns
        -------
        Tensor
            Heatmap of shape (1, 1, H, W).
        """
        heatmap = (weights * activations).sum(dim=1, keepdim=True)

        return torch.relu(heatmap)
    
    def _resize_heatmap(self, heatmap: Tensor, size: tuple[int, int]) -> Tensor:
        """
        Resize the Grad-CAM heatmap to the target spatial dimensions.
        """
        return F.interpolate(
            heatmap,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def _normalize_heatmap(self, heatmap: Tensor) -> Tensor:
        """
        Normalize the heatmap to the range [0, 1].
        """
        heatmap = heatmap.squeeze(0).squeeze(0)

        minimum = heatmap.min()
        maximum = heatmap.max()

        denominator = maximum - minimum

        if denominator.item() > 0:
            heatmap = (heatmap - minimum) / denominator
        else:
            heatmap = torch.zeros_like(heatmap)

        return heatmap

    def generate(self, image: Tensor, target_class: int | None = None) -> GradCAMResult:
        """
        Generate a Grad-CAM explanation.

        Parameters
        ----------
        image:
            Input image tensor of shape (C, H, W).
        target_class:
            Target class to explain. If None, explains the predicted class.
        """
        if image.ndim != 3:
            raise ValueError("Expected image tensor with shape (C, H, W).")

        # ---- Hook lifecycle ----
        height, width = image.shape[-2:]
        hook = ActivationHook(self._target_layer)

        was_training = self._model.training
        self._model.eval()

        try:
            # ---- Forward ----
            image = image.to(next(self._model.parameters()).device)
            image = image.unsqueeze(0)

            self._model.zero_grad(set_to_none=True)
            
            logits = self._model(image)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted_class = int(logits.argmax(dim=1).item())

            if target_class is None:
                target_class = predicted_class

            # ---- Backward ----

            num_classes = int(logits.shape[1])

            if not 0 <= target_class < num_classes:
                raise ValueError(f"target_class must be in [0, {num_classes - 1}].")
            
            score = logits[:, target_class]

            score.backward()

            # ---- Retrieve hook data ----
            activations = hook.activations
            gradients = hook.gradients

            # Sanity check (important for debugging)
            if activations.shape != gradients.shape:
                raise RuntimeError(
                    "Activations and gradients shape mismatch.",
                )

            weights = self._compute_weights(gradients)

            heatmap = self._compute_heatmap(
                activations,
                weights,
            )

            heatmap = self._resize_heatmap(
                heatmap,
                size=(height, width)
            )

            heatmap = self._normalize_heatmap(
                heatmap,
            )

        finally:
            hook.remove()
            self._model.train(was_training)

        heatmap = heatmap.detach().cpu().numpy().astype("float32")
        probabilities = (
            probabilities.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
        )

        return GradCAMResult(
            heatmap=heatmap,
            predicted_class=predicted_class,
            target_class=target_class,
            class_probabilities=probabilities,
            image_size=(height, width)
        )