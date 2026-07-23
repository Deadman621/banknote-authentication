# tests/test_end_to_end.py

from __future__ import annotations

from pathlib import Path

import torch

from src.application.experiment import ExperimentApplication
from src.application.inference import InferenceApplication
from src.application.evaluation import EvaluationApplication
from src.application.gradcam import GradCAMApplication


EXPERIMENT_ROOT = Path(
    "experiments/denomination/cnn/train_2/20260722_085918"
)

IMAGE_PATH = Path(
    "tests/inference/images/2.png"
)


def test_experiment_loads():

    app = ExperimentApplication()

    context = app.prepare(
        existing_run=EXPERIMENT_ROOT
    )

    assert context.module == "denomination"

    assert context.config is not None

    assert context.device in (
        torch.device("cpu"),
        torch.device("cuda"),
    )


def test_checkpoint_loads():

    experiment_app = ExperimentApplication()

    context = experiment_app.prepare(
        existing_run=EXPERIMENT_ROOT
    )

    checkpoint = (
        EXPERIMENT_ROOT
        / "checkpoints"
        / "best.pt"
    )

    model = experiment_app.load_checkpoint_model(
        context,
        checkpoint,
    )

    assert model is not None

    model.eval()


def test_prediction_pipeline():

    experiment_app = ExperimentApplication()
    inference_app = InferenceApplication()

    context = experiment_app.prepare(
        existing_run=EXPERIMENT_ROOT
    )

    checkpoint = (
        EXPERIMENT_ROOT
        / "checkpoints"
        / "best.pt"
    )

    model = inference_app.prepare_model(
        context,
        checkpoint,
    )

    result = inference_app.predict_with_module(
        context=context,
        model=model,
        image=IMAGE_PATH,
    )


    assert result is not None

    assert isinstance(
        result.predicted_class,
        int,
    )

    assert 0 <= result.confidence <= 1



def test_evaluation_pipeline():

    experiment_app = ExperimentApplication()
    evaluation_app = EvaluationApplication()

    context = experiment_app.prepare(
        existing_run=EXPERIMENT_ROOT
    )


    result = evaluation_app.evaluate(
        context=context,
        checkpoint_name="best.pt",
    )


    assert result is not None

    assert result.metrics is not None



def test_gradcam_pipeline(tmp_path):

    experiment_app = ExperimentApplication()
    gradcam_app = GradCAMApplication()


    context = experiment_app.prepare(
        existing_run=EXPERIMENT_ROOT
    )


    checkpoint = (
        EXPERIMENT_ROOT
        / "checkpoints"
        / "best.pt"
    )


    model = experiment_app.load_checkpoint_model(
        context,
        checkpoint,
    )


    result, overlay = gradcam_app.generate(
        context=context,
        model=model,
        image=IMAGE_PATH,
    )


    assert result is not None

    assert result.heatmap is not None

    assert overlay is not None



def test_full_pipeline():

    """
    Complete user workflow test.

    This is intentionally simple:
    
    train experiment
        |
        evaluate
        |
        predict image
        |
        generate explanation

    """

    experiment_app = ExperimentApplication()
    inference_app = InferenceApplication()
    evaluation_app = EvaluationApplication()
    gradcam_app = GradCAMApplication()


    context = experiment_app.prepare(
        existing_run=EXPERIMENT_ROOT
    )


    checkpoint = (
        EXPERIMENT_ROOT
        / "checkpoints"
        / "best.pt"
    )


    model = experiment_app.load_checkpoint_model(
        context,
        checkpoint,
    )


    prediction = inference_app.predict_with_module(
        context,
        model,
        IMAGE_PATH,
    )


    evaluation = evaluation_app.evaluate(
        context,
        "best.pt",
    )


    gradcam_result, overlay = gradcam_app.generate(
        context,
        model,
        IMAGE_PATH,
    )


    assert prediction.confidence > 0

    assert evaluation.metrics.accuracy >= 0

    assert gradcam_result.heatmap.shape[0] > 0

    assert overlay.shape[0] > 0