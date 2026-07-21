from __future__ import annotations

import argparse
from pathlib import Path

from scripts._common import (
    load_experiment_bundle,
    predict_image,
    save_prediction_results,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one or more images.")
    parser.add_argument(
        "--experiment-root",
        required=True,
        help="Path to an experiment directory.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image file(s) or directory/directories containing images.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Checkpoint filename inside the experiment checkpoint directory.",
    )
    return parser.parse_args()


def collect_images(paths: list[str]) -> list[Path]:
    images: list[Path] = []

    for path_str in paths:
        path = Path(path_str)

        if not path.exists():
            print(f"Warning: {path} does not exist. Skipping.")
            continue

        if path.is_file():
            images.append(path)
        elif path.is_dir():
            images.extend(
                sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
            )

    return images


def main() -> None:
    args = parse_args()

    bundle = load_experiment_bundle(
        Path(args.experiment_root),
        args.checkpoint_name,
    )

    image_paths = collect_images(args.images)

    if not image_paths:
        raise ValueError("No valid images found.")

    results: list[dict[str, object]] = []

    for image_path in image_paths:
        result, _ = predict_image(
            bundle.model,
            image_path,
            bundle.config.dataset.image_size,
        )

        predicted_name = (
            bundle.class_names[result.predicted_class]
            if result.predicted_class < len(bundle.class_names)
            else str(result.predicted_class)
        )

        print(f"{image_path}: {predicted_name} ({result.confidence:.4f})")

        results.append(
            {
                "image": str(image_path),
                "predicted_class": result.predicted_class,
                "predicted_label": predicted_name,
                "confidence": result.confidence,
                "probabilities": result.probabilities.tolist(),
            }
        )

    save_prediction_results(
        Path(args.experiment_root) / "predictions" / "predictions.json",
        results,
    )


if __name__ == "__main__":
    main()