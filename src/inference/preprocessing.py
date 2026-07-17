from pathlib import Path
from PIL import Image
from typing import cast

from torch import Tensor
from torchvision import transforms
from src.inference.state import PreprocessingConfig

def _load_image(image_path: Path) -> Image.Image:
    """Load an image from disk and convert it to RGB."""

    with Image.open(image_path) as image:
        return image.convert("RGB")
    
def _build_transform(config: PreprocessingConfig) -> transforms.Compose:
    """Build the preprocessing pipeline.

    Args:
        config: Preprocessing configuration.

    Returns:
        A torchvision transform pipeline.
    """

    return transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.mean,
                std=config.std,
            ),
        ]
    )

def _apply_transform(image: Image.Image, transform: transforms.Compose) -> Tensor:
    """Apply a preprocessing pipeline to an image."""

    return cast(Tensor, transform(image))


def preprocess_image( image_path: Path, config: PreprocessingConfig) -> Tensor:
    """Preprocess an image for inference."""

    image = _load_image(image_path)
    transform = _build_transform(config)
    tensor = _apply_transform(image, transform)

    return tensor.unsqueeze(0)
