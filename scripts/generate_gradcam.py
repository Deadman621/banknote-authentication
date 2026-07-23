# scripts/generate_gradcam.py

from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np
from PIL import Image

from src.application.experiment import ExperimentApplication
from src.application.gradcam import GradCAMApplication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and save a Grad-CAM overlay."
    )

    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Path to experiment directory.",
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Image path to explain.",
    )

    parser.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Checkpoint filename inside checkpoints directory.",
    )

    parser.add_argument(
        "--target-layer",
        default=None,
        help="Optional dotted path to target layer.",
    )

    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Optional class index to explain.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiment_app = ExperimentApplication()
    gradcam_app = GradCAMApplication()

    context = experiment_app.prepare(
        existing_run=args.experiment_root,
    )

    checkpoint = (
        args.experiment_root
        / "checkpoints"
        / args.checkpoint_name
    )

    model = experiment_app.load_checkpoint_model(
        context=context,
        checkpoint=checkpoint,
    )

    result, overlay = gradcam_app.generate(
        context=context,
        model=model,
        image=args.image,
        target_layer=args.target_layer,
        target_class=args.target_class,
    )

    output_path = (
        args.output
        if args.output is not None
        else (
            args.experiment_root
            / "gradcam"
            / f"{args.image.stem}_gradcam.png"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        np.asarray(
            overlay,
            dtype=np.uint8,
        )
    ).save(
        output_path,
    )

    print(output_path)

    print(
        f"predicted_class={result.predicted_class} "
        f"target_class={result.target_class}"
    )


if __name__ == "__main__":
    main()