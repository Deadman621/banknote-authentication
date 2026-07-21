#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from tqdm import tqdm

from src.core.experiment import load_preprocessing_configs
from src.preprocessing.preprocess import Preprocessor
from src.utils.image import save_image

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess an image dataset.")

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input dataset directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset directory.",
    )

    return parser.parse_args()


def iter_images(root: Path) -> Iterator[Path]:
    """Yield all supported image files recursively."""

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def preprocess_dataset(input_dir: Path, output_dir: Path, preprocessor: Preprocessor) -> None:
    images = list(iter_images(input_dir))

    processed_count = 0
    failed_count = 0

    for image_path in tqdm(images, desc="Preprocessing"):
        try:
            processed = preprocessor.preprocess_image(image_path)

            relative = image_path.relative_to(input_dir)
            destination = output_dir / relative

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_image(
                processed,
                destination.stem,
                destination.parent,
            )

            processed_count += 1

        except Exception as exc:
            failed_count += 1
            print(f"[ERROR] {image_path}: {exc}")

    print(
        f"\nCompleted preprocessing.\n"
        f"Processed: {processed_count}\n"
        f"Failed:    {failed_count}"
    )


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {args.input}"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    preprocessing_config, logging_config = load_preprocessing_configs()

    preprocessor = Preprocessor(
        preprocessing_config,
        logging_config,
    )

    preprocess_dataset(
        input_dir=args.input,
        output_dir=args.output,
        preprocessor=preprocessor,
    )


if __name__ == "__main__":
    main()