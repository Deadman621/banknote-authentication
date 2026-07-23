# scripts/predict.py

from __future__ import annotations

import argparse
from pathlib import Path

from src.application.experiment import ExperimentApplication
from src.application.inference import InferenceApplication


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on a trained experiment."
    )

    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Path to experiment directory.",
    )

    parser.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Checkpoint filename inside checkpoints directory.",
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Input image path or directory containing images.",
    )

    return parser.parse_args()


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    if path.is_dir():
        return sorted(
            image
            for image in path.iterdir()
            if image.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    raise FileNotFoundError(
        f"Image path does not exist: {path}"
    )


def main() -> None:
    args = parse_args()

    experiment_app = ExperimentApplication()
    inference_app = InferenceApplication()

    context = experiment_app.prepare(
        existing_run=args.experiment_root,
    )

    checkpoint = (
        args.experiment_root
        / "checkpoints"
        / args.checkpoint_name
    )

    model = inference_app.prepare_model(
        context=context,
        checkpoint=checkpoint,
    )

    images = collect_images(
        args.image,
    )

    if not images:
        raise ValueError(
            "No supported images found."
        )

    class_names = getattr(
        context.config.dataset,
        "class_names",
        None,
    )

    for image_path in images:
        result = inference_app.predict_with_module(
            context=context,
            model=model,
            image=image_path,
        )

        if class_names is not None:
            prediction = class_names[
                result.predicted_class
            ]
        else:
            prediction = str(
                result.predicted_class
            )

        print(
            f"Image: {image_path.name}"
        )

        print(
            f"Prediction: {prediction}"
        )

        print(
            f"Confidence: {result.confidence:.4f}"
        )

        print("-" * 40)


if __name__ == "__main__":
    main()