"""Advanced Scoring Methods - Weighted Average and Gaussian Smoothing"""

import numpy as np
from .step3_scoring import ScoringMethod


class WeightedAverageScoring(ScoringMethod):
    """Gaussian weighted average - center of window gets more weight

    Uses Gaussian weights so that points at the center of windows
    contribute more to the final score than points at the edges.
    """

    def score(self,
              subsequence_scores: np.ndarray,
              window_size: int,
              stride: int,
              original_length: int) -> np.ndarray:
        """Weighted average aggregation with Gaussian weights"""
        # Create Gaussian weights centered at middle of window
        center = window_size / 2
        sigma = window_size / 6
        positions = np.arange(window_size)
        weights = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
        weights = weights / weights.sum()

        point_scores = np.zeros(original_length)
        weight_sums = np.zeros(original_length)

        for i, score in enumerate(subsequence_scores):
            start = i * stride
            end = min(start + window_size, original_length)
            actual_window = end - start

            # Use appropriate weights for actual window size
            if actual_window < window_size:
                window_weights = weights[:actual_window]
            else:
                window_weights = weights

            # Weighted sum
            point_scores[start:end] += score * window_weights
            weight_sums[start:end] += window_weights

        # Average (avoid division by zero)
        point_scores = point_scores / np.maximum(weight_sums, 1e-8)

        return point_scores


class GaussianSmoothingScoring(ScoringMethod):
    """Apply Gaussian smoothing after aggregation

    First does average pooling, then applies Gaussian smoothing
    to produce smooth score transitions.

    Args:
        sigma: Smoothing strength (higher = more smoothing)
    """

    def __init__(self, sigma: float = 2.0):
        self.sigma = sigma

    def score(self,
              subsequence_scores: np.ndarray,
              window_size: int,
              stride: int,
              original_length: int) -> np.ndarray:
        """Average pooling followed by Gaussian smoothing"""
        from scipy.ndimage import gaussian_filter1d
        from .step3_scoring import AveragePoolingScoring

        # First do average pooling
        avg_scoring = AveragePoolingScoring()
        point_scores = avg_scoring.score(subsequence_scores, window_size, stride, original_length)

        # Apply Gaussian smoothing
        smoothed = gaussian_filter1d(point_scores, sigma=self.sigma)

        return smoothed
