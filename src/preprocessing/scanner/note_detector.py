import numpy as np
import logging

from cv2.typing import MatLike
from pathlib import Path
from numpy.typing import NDArray

from src.utils.logger import get_logger
from src.preprocessing.scanner.boundary_detector import BoundaryDetector
from src.preprocessing.scanner.perspective_corrector import PerspectiveCorrector
from src.core.config import ExperimentConfig
from src.utils.image import (
    save_debug_image,
    draw_contour,
    load_image,
    resize_max_dim,
)

class NoteDetector:
    """Full CamScanner-style pipeline: detect note → warp to aligned rectangle."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.log = self.log = get_logger(__name__, level=getattr(logging, config.logging.level.upper()))
        self.resize_max = config.preprocessing.resize_max_dim
        self.boundary = BoundaryDetector(config)
        self.perspective = PerspectiveCorrector(config)
        self.debug = config.preprocessing.debug
        self.debug_dir = config.preprocessing.debug_output_dir

    def process(self, image_or_path: str | Path | NDArray[np.uint8]) -> MatLike | None:
        """Detect the note and return an aligned top-down view."""
        # Load image (or accept a numpy array directly)
        if isinstance(image_or_path, (str, Path)):
            image: NDArray[np.uint8] = load_image(str(image_or_path))
        else:
            image = image_or_path

        # 1. Resize for speed
        resized, _ = resize_max_dim(image, self.resize_max)
        if self.debug:
            save_debug_image(resized, "01_resized", self.debug_dir)

        # 2. Detect the 4 corners of the note
        corners = self.boundary.detect(resized)
        if corners is None:
            self.log.warning("No note boundary detected.")
            return None
        if self.debug:
            vis = draw_contour(resized, corners.reshape(-1, 1, 2))
            save_debug_image(vis, "02_boundary", self.debug_dir)

        # 3. Perspective-correct to a fixed top-down rectangle
        aligned = self.perspective.correct_auto_orient(resized, corners)
        if self.debug:
            save_debug_image(aligned, "03_aligned", self.debug_dir)

        return aligned