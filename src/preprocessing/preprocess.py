from pathlib import Path
from typing import Optional

import numpy as np

from .scanner.note_detector import NoteDetector
from src.core.config import LoggingConfig, PreprocessingConfig
from src.utils.image import load_image


class Preprocessor:
    def __init__(self, config: PreprocessingConfig, logging: LoggingConfig) -> None:
        self.note_detector = NoteDetector(config, logging)

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        original = load_image(image_path)
        processed = self.note_detector.process(original)

        return original if processed is None else np.asarray(processed)