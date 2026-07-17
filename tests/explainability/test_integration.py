from __future__ import annotations

import torch
from numpy.typing import NDArray
from torch import Tensor
from torch import nn

from src.explainability.gradcam import GradCAM
from src.explainability.visualization import overlay_heatmap


class TinyCNN(nn.Module):
    """
    Small CNN for Grad-CAM integration testing.
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=4,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(
            4 * 8 * 8,
            2,
        )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        features = self.features(images)

        flattened = torch.flatten(
            features,
            start_dim=1,
        )

        return self.classifier(flattened)


def test_gradcam_visualization_pipeline() -> None:
    """
    Test complete explainability pipeline.

    Flow:
        Tensor image
            |
            v
        GradCAM.generate()
            |
            v
        GradCAMResult
            |
            v
        overlay_heatmap()
            |
            v
        RGB visualization
    """

    torch.manual_seed(42)

    model = TinyCNN()

    target_layer = model.features[0]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    image = torch.rand(
        3,
        8,
        8,
        requires_grad=False,
    )

    result = gradcam.generate(
        image,
    )

    assert result.heatmap.shape == (
        8,
        8,
    )

    assert result.class_probabilities.shape == (
        2,
    )

    assert result.image_size == (
        8,
        8,
    )

    rgb_image: NDArray = (
        image.permute(
            1,
            2,
            0,
        )
        .mul(255)
        .byte()
        .numpy()
    )

    overlay = overlay_heatmap(
        rgb_image,
        result.heatmap,
    )

    assert overlay.shape == (
        8,
        8,
        3,
    )

    assert overlay.dtype.name == "uint8"