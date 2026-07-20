"""
Reusable image-transform pipelines for currency-recognition modules.
"""
from typing import cast

from collections.abc import Callable

from PIL import Image
from torch import Tensor
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def resize_and_pad(image: Image.Image, size: int) -> Image.Image:
    """
    Resize while preserving aspect ratio, then pad to square.
    """
    width, height = image.size

    scale = size / max(width, height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    pad_left = (size - new_width) // 2
    pad_top = (size - new_height) // 2
    
    padded = Image.new(image.mode, (size, size), color=0)
    padded.paste(image, (pad_left, pad_top))

    return padded

def build_train_transforms(image_size: int = 224) -> Callable[[Image.Image], Tensor]:
    """
    Create the training transform pipeline.

    Includes mild geometric and color augmentation suitable for
    currency-image classification.
    """
    if image_size <= 0:
        raise ValueError("image_size must be greater than 0.")

    return transforms.Compose(
        [
            transforms.Lambda(lambda img: resize_and_pad(img, image_size)),
            transforms.RandomRotation(degrees=8),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def build_eval_transforms(image_size: int = 224) -> Callable[[Image.Image], Tensor]:
    """
    Create deterministic transforms for validation and testing.
    """
    if image_size <= 0:
        raise ValueError("image_size must be greater than 0.")

    return transforms.Compose(
        [
            transforms.Lambda(lambda img: resize_and_pad(img, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )