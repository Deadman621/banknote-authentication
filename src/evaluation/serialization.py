"""
Serialization utilities for evaluation results.
"""

from __future__ import annotations
from src.evaluation.state import EvaluationResult

def serialize(result: EvaluationResult) -> dict[str, object]:
    """
    Convert an EvaluationResult into a JSON-serializable dictionary.

    Args:
        result:
            Evaluation result.

    Returns:
        Dictionary suitable for JSON serialization.
    """

    return {
        "loss": result.loss,
        "metrics": {
            "accuracy": result.metrics.accuracy,
            "precision": result.metrics.precision,
            "recall": result.metrics.recall,
            "f1": result.metrics.f1,
            "roc_auc": result.metrics.roc_auc,
        },
        "predictions": result.predictions.tolist(),
        "targets": result.targets.tolist(),
        "probabilities": result.probabilities.tolist(),
        "confusion_matrix": result.confusion_matrix.tolist(),
    }