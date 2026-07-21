import cv2
import numpy as np
from pathlib import Path
from typing import Any

from src.utils.image import to_gray, adaptive_canny_thresholds
from src.core.config import PreprocessingConfig

class BoundaryDetector:
    """Find the largest quadrilateral in an image (assumed to be the note)."""

    def __init__(self, config: PreprocessingConfig) -> None:
        pp = config
        self.gaussian_kernel = pp.gaussian_kernel
        self.gaussian_sigma = pp.gaussian_sigma
        self.canny_adaptive = pp.canny.adaptive
        self.canny_lower = pp.canny.lower
        self.canny_upper = pp.canny.upper
        self.min_area_ratio = pp.contour.min_area_ratio
        self.approx_epsilon = pp.contour.approx_epsilon

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        """Return the 4 corner points of the note, or None if not found."""
        # 1. Convert to grayscale
        gray = to_gray(image)

        # 2. Blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray, self.gaussian_kernel, self.gaussian_sigma)

        # 3. Canny edge detection (adaptive thresholds by default)
        if self.canny_adaptive:
            lower, upper = adaptive_canny_thresholds(blurred)
        else:
            lower, upper = self.canny_lower, self.canny_upper
        edges = cv2.Canny(blurred, lower, upper)

        # 4. Dilate to close small gaps in the boundary
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        # 5. Find contours (outer only)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.min_area_ratio

        # 6. Search largest → smallest for the first valid quadrilateral
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            if area < min_area:
                break
            peri = cv2.arcLength(cnt, closed=True)
            approx = cv2.approxPolyDP(cnt, self.approx_epsilon * peri, closed=True)
            if len(approx) == 4:
                return approx.reshape(4, 2)

        # 7. Fallback: use minAreaRect on the largest contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= min_area:
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            return box.astype(int)

        return None