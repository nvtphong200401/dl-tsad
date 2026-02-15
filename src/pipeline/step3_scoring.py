"""Step 3: Scoring - Convert sub-sequence scores to point-wise scores"""

from abc import ABC, abstractmethod
import numpy as np


class ScoringMethod(ABC):
    """Base class for Step 3: Convert sub-sequence scores to point-wise scores

    Scoring methods aggregate window-level anomaly scores to point-level scores.
    """

    @abstractmethod
    def score(self,
              subsequence_scores: np.ndarray,
              window_size: int,
              stride: int,
              original_length: int) -> np.ndarray:
        """Convert sub-sequence scores to point-wise scores

        Args:
            subsequence_scores: Window-level scores (N,)
            window_size: Size of each window
            stride: Stride between windows
            original_length: Length of original time series

        Returns:
            Point-wise scores (T,)
        """
        pass


class MaxPoolingScoring(ScoringMethod):
    """Each point gets maximum score from all windows containing it

    This is conservative - emphasizes detecting anomalies (high recall).
    """

    def score(self,
              subsequence_scores: np.ndarray,
              window_size: int,
              stride: int,
              original_length: int) -> np.ndarray:
        """Max pooling aggregation"""
        point_scores = np.zeros(original_length)

        for i, score in enumerate(subsequence_scores):
            start = i * stride
            end = min(start + window_size, original_length)
            # Max pooling - take maximum
            point_scores[start:end] = np.maximum(point_scores[start:end], score)

        return point_scores


class AveragePoolingScoring(ScoringMethod):
    """Each point gets average score from all windows containing it

    This is balanced - averages contributions from overlapping windows.
    """

    def score(self,
              subsequence_scores: np.ndarray,
              window_size: int,
              stride: int,
              original_length: int) -> np.ndarray:
        """Average pooling aggregation"""
        point_scores = np.zeros(original_length)
        counts = np.zeros(original_length)

        for i, score in enumerate(subsequence_scores):
            start = i * stride
            end = min(start + window_size, original_length)
            # Sum scores
            point_scores[start:end] += score
            counts[start:end] += 1

        # Average (avoid division by zero)
        point_scores = point_scores / np.maximum(counts, 1)

        return point_scores
