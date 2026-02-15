"""Step 4: Post-processing - Threshold and extract anomalies"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import numpy as np


class ThresholdDetermination(ABC):
    """Base class for threshold determination"""

    @abstractmethod
    def find_threshold(self,
                      scores: np.ndarray,
                      labels: Optional[np.ndarray] = None) -> float:
        """Determine threshold for anomaly detection

        Args:
            scores: Point-wise anomaly scores (T,)
            labels: Optional ground truth labels (T,)

        Returns:
            Threshold value
        """
        pass


class PercentileThreshold(ThresholdDetermination):
    """Use percentile of scores as threshold

    Simple unsupervised approach - assumes anomalies are rare.
    """

    def __init__(self, percentile: float = 95.0):
        """Initialize percentile threshold

        Args:
            percentile: Percentile to use (e.g., 95.0 = 95th percentile)
        """
        self.percentile = percentile

    def find_threshold(self,
                      scores: np.ndarray,
                      labels: Optional[np.ndarray] = None) -> float:
        """Return percentile of scores"""
        return np.percentile(scores, self.percentile)


class F1OptimalThreshold(ThresholdDetermination):
    """Find threshold that maximizes F1 score on validation set

    Requires labeled validation data. Searches for optimal threshold.
    """

    def find_threshold(self,
                      scores: np.ndarray,
                      labels: Optional[np.ndarray] = None) -> float:
        """Find F1-optimal threshold"""
        if labels is None:
            raise ValueError("F1OptimalThreshold requires labels")

        best_f1 = 0
        best_threshold = 0

        # Try percentiles from 80 to 99.9
        for p in np.linspace(80, 99.9, 100):
            thresh = np.percentile(scores, p)
            preds = (scores > thresh).astype(int)

            # Compute F1
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        return best_threshold


class PostProcessor:
    """Complete post-processing pipeline

    Handles:
    1. Threshold determination
    2. Binary prediction
    3. Filtering short anomalies
    4. Merging close anomalies
    """

    def __init__(self,
                 threshold_method: ThresholdDetermination,
                 min_anomaly_length: int = 1,
                 merge_gap: int = 0):
        """Initialize post-processor

        Args:
            threshold_method: Method to determine threshold
            min_anomaly_length: Remove anomaly segments shorter than this
            merge_gap: Merge anomalies separated by this many points or less
        """
        self.threshold_method = threshold_method
        self.min_anomaly_length = min_anomaly_length
        self.merge_gap = merge_gap

    def process(self,
                scores: np.ndarray,
                labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Full post-processing pipeline

        Args:
            scores: Point-wise anomaly scores (T,)
            labels: Optional ground truth for threshold tuning (T,)

        Returns:
            predictions: Binary predictions (T,)
            threshold: Determined threshold value
        """
        # Step 1: Determine threshold
        threshold = self.threshold_method.find_threshold(scores, labels)

        # Step 2: Extract anomalies
        predictions = (scores > threshold).astype(int)

        # Step 3: Filter short anomalies
        if self.min_anomaly_length > 1:
            predictions = self._filter_short_anomalies(predictions)

        # Step 4: Merge close anomalies
        if self.merge_gap > 0:
            predictions = self._merge_close_anomalies(predictions)

        return predictions, threshold

    def _filter_short_anomalies(self, predictions: np.ndarray) -> np.ndarray:
        """Remove anomaly segments shorter than min_anomaly_length"""
        segments = self._get_anomaly_segments(predictions)
        filtered = np.zeros_like(predictions)

        for start, end in segments:
            if end - start >= self.min_anomaly_length:
                filtered[start:end] = 1

        return filtered

    def _merge_close_anomalies(self, predictions: np.ndarray) -> np.ndarray:
        """Merge anomalies separated by gap <= merge_gap"""
        segments = self._get_anomaly_segments(predictions)

        if len(segments) == 0:
            return predictions

        merged = []
        current_start, current_end = segments[0]

        for start, end in segments[1:]:
            if start - current_end <= self.merge_gap:
                # Merge with current segment
                current_end = end
            else:
                # Save current and start new segment
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        merged.append((current_start, current_end))

        # Convert back to binary array
        result = np.zeros_like(predictions)
        for start, end in merged:
            result[start:end] = 1

        return result

    def _get_anomaly_segments(self, predictions: np.ndarray) -> List[Tuple[int, int]]:
        """Extract anomaly segments as (start, end) tuples"""
        segments = []
        in_anomaly = False
        start = 0

        for i, val in enumerate(predictions):
            if val == 1 and not in_anomaly:
                start = i
                in_anomaly = True
            elif val == 0 and in_anomaly:
                segments.append((start, i))
                in_anomaly = False

        # Handle case where anomaly extends to end
        if in_anomaly:
            segments.append((start, len(predictions)))

        return segments
