"""
Model evaluation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.evaluation.confusion_matrix import compute_confusion_matrix
from src.evaluation.metrics import classification_metrics
from src.evaluation.state import EvaluationResult
from src.training.utils import move_batch_to_device

class Evaluator:
    """
    Evaluates a model on a dataset.
    """

    def __init__(self, model: nn.Module, loss_fn: nn.Module, device: torch.device) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

        self.model.to(device)

    def evaluate(self, dataloader: DataLoader) -> EvaluationResult:
        """
        Evaluate the model.

        Args:
            dataloader:
                Evaluation dataloader.

        Returns:
            EvaluationResult containing metrics and predictions.
        """

        if len(dataloader) == 0:
            raise ValueError("Evaluation dataloader must not be empty.")

        self.model.eval()

        running_loss = 0.0

        prediction_batches: list[Tensor] = []
        target_batches: list[Tensor] = []
        probability_batches: list[Tensor] = []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = move_batch_to_device(images, labels, self.device)

                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                running_loss += float(loss.item())

                prediction_batches.append(predictions.cpu())
                target_batches.append(labels.cpu())

                probability_batches.append(probabilities.cpu())

        predictions = torch.cat(prediction_batches, dim=0)
        targets = torch.cat(target_batches, dim=0)
        probabilities = torch.cat(probability_batches, dim=0)

        metrics = classification_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=probabilities,
        )

        matrix = compute_confusion_matrix(
            predictions=predictions,
            targets=targets,
        )

        return EvaluationResult(
            loss=running_loss / len(dataloader),
            metrics=metrics,
            predictions=predictions,
            targets=targets,
            probabilities=probabilities,
            confusion_matrix=matrix,
        )