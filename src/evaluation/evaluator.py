"""Evaluator for anomaly detection pipelines"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from .metrics import (
    compute_f1_score,
    compute_precision_recall,
    compute_point_adjusted_f1,
    compute_vus_pr
)


@dataclass
class EvaluationResult:
    """Evaluation results container"""
    f1: float
    precision: float
    recall: float
    pa_f1: float
    vus_pr: Optional[float] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = None


class Evaluator:
    """Evaluate anomaly detection pipeline results"""

    def evaluate(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 scores: np.ndarray) -> EvaluationResult:
        """Compute all evaluation metrics

        Args:
            y_true: Ground truth labels (T,)
            y_pred: Predicted labels (T,)
            scores: Anomaly scores (T,)

        Returns:
            EvaluationResult with all metrics
        """
        # Standard metrics
        f1 = compute_f1_score(y_true, y_pred)
        precision, recall = compute_precision_recall(y_true, y_pred)

        # Point-adjusted F1
        pa_f1 = compute_point_adjusted_f1(y_true, y_pred)

        # VUS-PR
        vus_pr = compute_vus_pr(y_true, scores)

        return EvaluationResult(
            f1=f1,
            precision=precision,
            recall=recall,
            pa_f1=pa_f1,
            vus_pr=vus_pr,
            latency_ms=0.0,
            metadata={}
        )
