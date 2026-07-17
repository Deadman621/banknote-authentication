"""
Grad-CAM visualization utilities.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import cast
from numpy.typing import NDArray

def validate_image(image: NDArray[np.uint8]) -> None:
    """
    Validate an RGB image.

    Parameters
    ----------
    image:
        RGB image with shape (H, W, 3).

    Raises
    ------
    ValueError
        If the image has an invalid shape or dtype.
    """
    if image.ndim != 3:
        raise ValueError("Expected image with shape (H, W, 3).")

    if image.shape[2] != 3:
        raise ValueError("Expected RGB image.")

    if image.dtype != np.uint8:
        raise ValueError("Image dtype must be uint8.")


def validate_heatmap( heatmap: NDArray[np.float32], image_shape: tuple[int, int]) -> None:
    """
    Validate a normalized Grad-CAM heatmap.

    Parameters
    ----------
    heatmap:
        Heatmap with shape (H, W).

    image_shape:
        Expected spatial dimensions.

    Raises
    ------
    ValueError
        If the heatmap is invalid.
    """
    if heatmap.ndim != 2:
        raise ValueError("Expected heatmap with shape (H, W).")

    if heatmap.dtype != np.float32:
        raise ValueError("Heatmap dtype must be float32.")

    if heatmap.shape != image_shape:
        raise ValueError("Heatmap size does not match image size.")


def apply_colormap(heatmap: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Apply the JET colormap to a normalized heatmap.

    Parameters
    ----------
    heatmap:
        Heatmap normalized to [0, 1].

    Returns
    -------
    NDArray[np.uint8]
        RGB heatmap of shape (H, W, 3).
    """
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)

    colored = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    rgb = cv2.cvtColor(
        colored,
        cv2.COLOR_BGR2RGB,
    )

    return cast(
        NDArray[np.uint8],
        rgb,
    )

def overlay_heatmap(image: NDArray[np.uint8], heatmap: NDArray[np.float32], alpha: float = 0.4) -> NDArray[np.uint8]:
    """
    Overlay a Grad-CAM heatmap onto an RGB image.

    Parameters
    ----------
    image:
        RGB image of shape (H, W, 3).

    heatmap:
        Normalized heatmap of shape (H, W).

    alpha:
        Heatmap blending factor in [0, 1].

    Returns
    -------
    NDArray[np.uint8]
        Overlay image of shape (H, W, 3).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in the range [0, 1].")

    validate_image(image)

    validate_heatmap(heatmap, image.shape[:2])

    colored = apply_colormap(heatmap)

    overlay = cv2.addWeighted(
        image,
        1.0 - alpha,
        colored,
        alpha,
        0.0,
    )

    return cast(
        NDArray[np.uint8],
        overlay,
    )