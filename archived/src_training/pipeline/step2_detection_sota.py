"""SOTA Detection Methods - Hybrid and Association Discrepancy"""

import numpy as np
from typing import Optional
from .step2_detection import DetectionMethod


class HybridDetection(DetectionMethod):
    """AER detection: combine reconstruction + bidirectional regression errors.

    Expects input format from AERProcessor: (N, 3)
        column 0 = reconstruction error
        column 1 = backward regression error
        column 2 = forward regression error

    Scoring modes (matching the Orion reference):
        "mult"  – multiply (both scaled to [1,2])  **default**
        "sum"   – weighted sum with lambda_rec
        "rec"   – reconstruction error only
        "reg"   – regression error only

    Args:
        comb:       Combination mode. Default "mult".
        lambda_rec: Weight for reconstruction in "sum" mode (0–1). Default 0.5.
    """

    def __init__(self, comb: str = "mult", lambda_rec: float = 0.5,
                 # Legacy aliases kept so old configs don't crash
                 alpha: float = None, beta: float = None):
        self.comb = comb
        self.lambda_rec = lambda_rec

    def fit(self, X_processed: np.ndarray, y: Optional[np.ndarray] = None):
        pass

    def detect(self, X_processed: np.ndarray) -> np.ndarray:
        """Compute hybrid anomaly score

        Args:
            X_processed: (N, 3) — [rec_error, reg_b_error, reg_f_error]

        Returns:
            Anomaly scores (N,)
        """
        rec_error = X_processed[:, 0]
        reg_error = (X_processed[:, 1] + X_processed[:, 2]) / 2

        if self.comb == "mult":
            rec_s = self._minmax_scale(rec_error, (1, 2))
            reg_s = self._minmax_scale(reg_error, (1, 2))
            scores = rec_s * reg_s
        elif self.comb == "sum":
            rec_s = self._minmax_scale(rec_error, (0, 1))
            reg_s = self._minmax_scale(reg_error, (0, 1))
            scores = self.lambda_rec * rec_s + (1 - self.lambda_rec) * reg_s
        elif self.comb == "rec":
            scores = rec_error
        elif self.comb == "reg":
            scores = reg_error
        else:
            scores = rec_error + reg_error

        return scores

    @staticmethod
    def _minmax_scale(x: np.ndarray, feature_range=(0, 1)) -> np.ndarray:
        """MinMax scale to the given range (handles constant arrays)."""
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-10:
            return np.full_like(x, feature_range[0])
        scaled = (x - x_min) / (x_max - x_min)
        return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]


class AssociationDiscrepancyDetection(DetectionMethod):
    """Anomaly Transformer detection using association discrepancy.

    Input is the per-timestep anomaly scores from AnomalyTransformerProcessor.
    Shape: (N, W) — one score per timestep per window.
    """

    def fit(self, X_processed: np.ndarray, y: Optional[np.ndarray] = None):
        # No additional training needed
        pass

    def detect(self, X_processed: np.ndarray) -> np.ndarray:
        """Aggregate per-timestep anomaly scores to get one score per window.

        Args:
            X_processed: Per-timestep anomaly scores (N, W)

        Returns:
            Anomaly scores (N,)
        """
        # Mean anomaly score across the window
        scores = np.mean(X_processed, axis=1)
        return scores
