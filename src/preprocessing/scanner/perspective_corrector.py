import cv2
import numpy as np

from typing import Any
from numpy.typing import NDArray

from src.utils.image import order_points
from src.core.config import PreprocessingConfig

from cv2.typing import MatLike

class PerspectiveCorrector:
    """Warp a quadrilateral region to a fixed-size rectangle."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.out_w = config.aligned_output.width
        self.out_h = config.aligned_output.height

    def correct(self, image: MatLike, corners: NDArray[np.floating[Any] | np.integer[Any]]) -> MatLike:
        """Return a top-down warped view of the region defined by `corners`."""
        # 1. Order the 4 corner points consistently (TL, TR, BR, BL)
        rect = order_points(corners)

        # 2. Define the destination rectangle (fixed output size)
        dst = np.array([
            [0, 0],
            [self.out_w - 1, 0],
            [self.out_w - 1, self.out_h - 1],
            [0, self.out_h - 1],
        ], dtype="float32")

        # 3. Compute perspective transform matrix and apply
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (self.out_w, self.out_h))
        return warped

    def correct_auto_orient(self, image: MatLike, corners: NDArray[np.floating[Any] | np.integer[Any]]) -> MatLike:
        """Warp, then rotate 90° if the source region is portrait (taller than wide).

        BDT notes are landscape; if the warped result comes out portrait we rotate.
        """
        warped = self.correct(image, corners)
        rect = order_points(corners)
        top_w = np.linalg.norm(rect[1] - rect[0])
        left_h = np.linalg.norm(rect[3] - rect[0])
        if left_h > top_w:  # source was portrait — rotate the warped result
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            warped = cv2.resize(
                warped,
                (self.out_w, self.out_h),
                interpolation=cv2.INTER_AREA,
            )
        return warped