"""SOTA Detection Methods - Hybrid and Association Discrepancy"""

import numpy as np
from typing import Optional
from .step2_detection import DetectionMethod


class HybridDetection(DetectionMethod):
    """AER-style: Combine reconstruction + prediction + bidirectional

    Expects input format: [original, recon, pred_f, pred_b]
    each of shape (N, W*D)

    Args:
        alpha: Weight between reconstruction and prediction errors (0-1)
        beta: Weight between forward and backward prediction (0-1)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta

    def fit(self, X_processed: np.ndarray, y: Optional[np.ndarray] = None):
        # No training needed for detection step with AER
        pass

    def detect(self, X_processed: np.ndarray) -> np.ndarray:
        """Compute hybrid anomaly score

        Args:
            X_processed: Concatenated [original, recon, pred_f, pred_b] (N, 4*W*D)

        Returns:
            Anomaly scores (N,)
        """
        # Split input into 4 parts
        n_samples, total_dim = X_processed.shape
        chunk_size = total_dim // 4

        original = X_processed[:, :chunk_size]
        recon = X_processed[:, chunk_size:2*chunk_size]
        pred_f = X_processed[:, 2*chunk_size:3*chunk_size]
        pred_b = X_processed[:, 3*chunk_size:]

        # Reconstruction error
        recon_error = np.mean((original - recon) ** 2, axis=1)

        # Forward prediction error
        pred_error_f = np.mean((original - pred_f) ** 2, axis=1)

        # Backward prediction error
        pred_error_b = np.mean((original - pred_b) ** 2, axis=1)

        # Bidirectional prediction error
        pred_error = self.beta * pred_error_f + (1 - self.beta) * pred_error_b

        # Combined score
        scores = self.alpha * recon_error + (1 - self.alpha) * pred_error

        return scores


class AssociationDiscrepancyDetection(DetectionMethod):
    """Anomaly Transformer detection using association discrepancy

    Input is the discrepancy computed by AnomalyTransformerProcessor
    Shape: (N, W*W)
    """

    def fit(self, X_processed: np.ndarray, y: Optional[np.ndarray] = None):
        # No additional training needed
        pass

    def detect(self, X_processed: np.ndarray) -> np.ndarray:
        """Aggregate discrepancy to get anomaly score per window

        Args:
            X_processed: Association discrepancy (N, W*W)

        Returns:
            Anomaly scores (N,)
        """
        # Compute score as mean discrepancy per window
        scores = np.mean(X_processed, axis=1)
        return scores
