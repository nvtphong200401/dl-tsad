"""Evaluation metrics for anomaly detection"""

import numpy as np
from typing import Tuple, List


def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard F1 score

    Args:
        y_true: Ground truth labels (T,)
        y_pred: Predicted labels (T,)

    Returns:
        F1 score
    """
    precision, recall = compute_precision_recall(y_true, y_pred)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute precision and recall

    Args:
        y_true: Ground truth labels (T,)
        y_pred: Predicted labels (T,)

    Returns:
        (precision, recall)
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


def compute_point_adjusted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Point-adjusted F1 score

    For each anomaly segment in ground truth:
    - If ANY point in the segment is detected, count as TP
    - Otherwise, count as FN

    For each predicted anomaly segment:
    - If it overlaps with ANY ground truth segment, count as TP
    - Otherwise, count as FP

    This is more lenient than point-wise F1 for continuous anomalies.

    Reference: "Towards a Rigorous Evaluation of Time-Series Anomaly Detection" (AAAI 2022)

    Args:
        y_true: Ground truth labels (T,)
        y_pred: Predicted labels (T,)

    Returns:
        Point-adjusted F1 score
    """
    # Get anomaly segments
    true_segments = _get_segments(y_true)
    pred_segments = _get_segments(y_pred)

    if len(true_segments) == 0:
        # No ground truth anomalies
        if len(pred_segments) == 0:
            return 1.0  # Perfect - no false positives
        else:
            return 0.0  # All false positives

    if len(pred_segments) == 0:
        # No predictions but have ground truth
        return 0.0  # All false negatives

    # Count TPs for recall (ground truth perspective)
    tp_recall = 0
    for true_start, true_end in true_segments:
        # Check if any prediction overlaps with this true segment
        detected = False
        for pred_start, pred_end in pred_segments:
            if _segments_overlap(true_start, true_end, pred_start, pred_end):
                detected = True
                break
        if detected:
            tp_recall += 1

    # Count TPs for precision (prediction perspective)
    tp_precision = 0
    for pred_start, pred_end in pred_segments:
        # Check if this prediction overlaps with any true segment
        correct = False
        for true_start, true_end in true_segments:
            if _segments_overlap(pred_start, pred_end, true_start, true_end):
                correct = True
                break
        if correct:
            tp_precision += 1

    # Compute precision and recall
    precision = tp_precision / len(pred_segments)
    recall = tp_recall / len(true_segments)

    # Compute F1
    if precision + recall == 0:
        return 0.0

    pa_f1 = 2 * precision * recall / (precision + recall)
    return pa_f1


def _get_segments(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Extract anomaly segments as (start, end) tuples

    Args:
        labels: Binary labels (T,)

    Returns:
        List of (start, end) tuples
    """
    segments = []
    in_anomaly = False
    start = 0

    for i, val in enumerate(labels):
        if val == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif val == 0 and in_anomaly:
            segments.append((start, i))
            in_anomaly = False

    # Handle case where anomaly extends to end
    if in_anomaly:
        segments.append((start, len(labels)))

    return segments


def _segments_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check if two segments overlap

    Args:
        start1, end1: First segment [start1, end1)
        start2, end2: Second segment [start2, end2)

    Returns:
        True if segments overlap
    """
    return not (end1 <= start2 or end2 <= start1)


def compute_vus_pr(y_true: np.ndarray, scores: np.ndarray, num_thresholds: int = 200) -> float:
    """Volume Under Surface for Precision-Recall curve

    More reliable metric than point-wise metrics according to TSB-AD benchmark.
    Computes precision-recall curve with sliding threshold and integrates the area.

    Reference: TSB-AD (NeurIPS 2024)

    Args:
        y_true: Binary labels (T,)
        scores: Anomaly scores (T,)
        num_thresholds: Number of thresholds to evaluate

    Returns:
        VUS-PR score between 0 and 1 (higher is better)
    """
    # Handle edge cases
    if len(scores) == 0 or len(y_true) == 0:
        return 0.0

    if y_true.sum() == 0:
        # No anomalies in ground truth
        return 1.0 if scores.sum() == 0 else 0.0

    # Get range of thresholds
    min_score = scores.min()
    max_score = scores.max()

    if min_score == max_score:
        # All scores are the same
        return 0.5

    thresholds = np.linspace(min_score, max_score, num_thresholds)

    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)

        # Compute precision and recall
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    # Convert to numpy arrays
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Sort by recall for proper area computation
    sorted_indices = np.argsort(recalls)
    recalls_sorted = recalls[sorted_indices]
    precisions_sorted = precisions[sorted_indices]

    # Compute area under PR curve using trapezoidal rule
    vus_pr = np.trapezoid(precisions_sorted, recalls_sorted)

    # Normalize to [0, 1] (recall goes from 0 to 1)
    vus_pr = max(0.0, min(1.0, vus_pr))

    return vus_pr
