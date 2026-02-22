"""Step 2: Detection Method - Compute anomaly scores for windows"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List
import numpy as np
from sklearn.neighbors import NearestNeighbors

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from evidence import StatisticalEvidenceExtractor


class DetectionMethod(ABC):
    """Base class for Step 2: Detection

    Detection methods take processed windows and compute anomaly scores
    for each window (sub-sequence level).
    """

    @abstractmethod
    def fit(self, X_processed: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit detection method on processed training data

        Args:
            X_processed: Processed windows (N, W, D')
            y: Optional labels (N,)
        """
        pass

    @abstractmethod
    def detect(self, X_processed: np.ndarray) -> np.ndarray:
        """Return sub-sequence level anomaly scores

        Args:
            X_processed: Processed windows (N, W, D')

        Returns:
            Anomaly scores (N,) - one score per window
        """
        pass


class DistanceBasedDetection(DetectionMethod):
    """Distance-based detection using K-Nearest Neighbors

    Stores normal training data and computes anomaly score as
    the average distance to k-nearest neighbors.
    """

    def __init__(self, k: int = 5, method: str = "knn"):
        """Initialize distance-based detection

        Args:
            k: Number of nearest neighbors
            method: "knn" (K-Nearest Neighbors)
        """
        self.k = k
        self.method = method
        self.train_data = None
        self.nbrs = None

    def fit(self, X_processed: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Store training data for distance comparison

        Args:
            X_processed: Processed windows (N, W, D')
            y: Optional labels (ignored for unsupervised)
        """
        # Flatten windows to vectors for distance computation
        N, W, D = X_processed.shape
        self.train_data = X_processed.reshape(N, -1)  # (N, W*D)

        # Fit k-NN model
        self.nbrs = NearestNeighbors(n_neighbors=self.k)
        self.nbrs.fit(self.train_data)

    def detect(self, X_processed: np.ndarray) -> np.ndarray:
        """Compute distance to k-nearest neighbors as anomaly score

        Args:
            X_processed: Processed windows (N, W, D')

        Returns:
            Anomaly scores (N,) - average distance to k-NN
        """
        if self.nbrs is None:
            raise ValueError("Must call fit() before detect()")

        # Flatten windows
        N, W, D = X_processed.shape
        X_flat = X_processed.reshape(N, -1)

        # Compute distances to k-nearest neighbors
        distances, _ = self.nbrs.kneighbors(X_flat)

        # Anomaly score = average distance to k-NN
        scores = np.mean(distances, axis=1)

        return scores


class EvidenceBasedDetection(DetectionMethod):
    """Detection via statistical evidence extraction.

    Uses foundation model forecasts and statistical tests to produce
    anomaly scores. Implements DetectionMethod for backward compatibility:
    detect() returns np.ndarray (N,).

    Full evidence dictionaries are stored as self.evidence_results
    for consumption by downstream LLM reasoning (Week 3).
    """

    def __init__(
        self,
        enabled_categories: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        config: Optional[Dict] = None
    ):
        self.extractor = StatisticalEvidenceExtractor(
            enabled_categories=enabled_categories,
            config=config
        )

        # Default aggregation weights for collapsing evidence to scalar
        self.weights = weights or {
            'mae': 0.10,
            'mse': 0.10,
            'mape': 0.05,
            'violation_ratio': 0.08,
            'mean_surprise': 0.07,
            'max_abs_z_score': 0.10,
            'grubbs_statistic': 0.08,
            'max_cusum': 0.07,
            'kl_divergence': 0.08,
            'normalized_wasserstein': 0.07,
            'max_acf_diff': 0.05,
            'volatility_ratio': 0.08,
            'slope_diff': 0.07,
        }

        # State set during fit()
        self.train_data_sample = None
        self.training_baselines = None

        # State set via set_forecast_context()
        self.forecast_results = None
        self.forecast_train_statistics = None

        # Output state after detect()
        self.evidence_results = None

    def set_forecast_context(
        self,
        forecast_results: List[Dict],
        train_statistics: Dict
    ) -> None:
        """Inject forecast context from Step 1.

        Called by the orchestrator between Step 1 and Step 2.

        Args:
            forecast_results: List of forecast dicts from FoundationModelProcessor
            train_statistics: Training statistics dict from FoundationModelProcessor
        """
        self.forecast_results = forecast_results
        self.forecast_train_statistics = train_statistics

    def fit(self, X_processed: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Store training baselines for normalization.

        Args:
            X_processed: Training windows (N, W, D)
            y: Optional labels (ignored)
        """
        N, W, D = X_processed.shape

        # Store representative training data (flattened, subsampled)
        all_values = X_processed[:, :, 0].flatten()
        max_samples = 10000
        if len(all_values) > max_samples:
            indices = np.random.choice(len(all_values), max_samples, replace=False)
            self.train_data_sample = all_values[indices]
        else:
            self.train_data_sample = all_values.copy()

        # Compute training baselines by running evidence on a sample of training windows
        n_baseline = min(50, N)
        baseline_indices = np.random.choice(N, n_baseline, replace=False)

        baseline_metrics = []
        train_mean = float(np.mean(self.train_data_sample))
        train_std = float(np.std(self.train_data_sample))

        for idx in baseline_indices:
            window = X_processed[idx, :, 0]
            metrics = {}

            z_scores = np.abs((window - train_mean) / (train_std + 1e-10))
            metrics['max_abs_z_score'] = float(np.max(z_scores))
            metrics['volatility_ratio'] = float(np.std(window) / (train_std + 1e-10))
            metrics['mae'] = float(np.mean(np.abs(window - train_mean)))
            metrics['mse'] = float(np.mean((window - train_mean) ** 2))

            baseline_metrics.append(metrics)

        # Store baselines as percentiles for normalization
        self.training_baselines = {}
        if baseline_metrics:
            for key in baseline_metrics[0]:
                values = [m.get(key, 0) for m in baseline_metrics]
                self.training_baselines[key] = {
                    'p50': float(np.percentile(values, 50)),
                    'p95': float(np.percentile(values, 95)),
                    'max': float(np.max(values))
                }

    def detect(self, X_processed: np.ndarray) -> np.ndarray:
        """Extract evidence and aggregate to scalar scores.

        Args:
            X_processed: Test windows (N, W, D)

        Returns:
            Anomaly scores (N,) - one scalar per window
        """
        N, W, D = X_processed.shape
        scores = np.zeros(N)
        self.evidence_results = []

        # Build train_statistics: use forecast context if available,
        # otherwise compute from stored training data sample
        train_stats = self.forecast_train_statistics
        if (not train_stats or 'mean' not in train_stats) and self.train_data_sample is not None:
            train_stats = {
                'mean': float(np.mean(self.train_data_sample)),
                'std': float(np.std(self.train_data_sample))
            }

        for i in range(N):
            window = X_processed[i, :, 0]

            # Get forecast for this window (if available)
            # Fix: align forecast with actual future values, not the context window.
            # Window i covers positions [i, i+W-1]. The forecast predicts positions
            # [i+W, i+W+H-1]. With stride=1, window i+W starts at position i+W,
            # so X_processed[i+W, :, 0] gives actual values at [i+W, i+2W-1].
            forecast_result = None
            if self.forecast_results is not None and i < len(self.forecast_results):
                fr = self.forecast_results[i]
                if fr.get('forecast') is not None and (i + W) < N:
                    H = len(fr['forecast'])
                    # Get actual future values from subsequent windows
                    actual_future = X_processed[i + W, :min(H, W), 0]
                    forecast_result = {
                        **fr,
                        '_actual_future': actual_future,
                    }
                else:
                    forecast_result = fr

            # Extract evidence — pass actual future for forecast comparison
            actual_for_forecast = None
            if forecast_result is not None and '_actual_future' in forecast_result:
                actual_for_forecast = forecast_result.pop('_actual_future')

            evidence = self.extractor.extract(
                test_window=window,
                forecast_result=forecast_result,
                train_statistics=train_stats,
                train_data=self.train_data_sample,
                actual_future=actual_for_forecast
            )

            evidence['window_index'] = i
            self.evidence_results.append(evidence)

            # Aggregate to scalar score
            scores[i] = self._aggregate_evidence(evidence)

        return scores

    def _aggregate_evidence(self, evidence: Dict) -> float:
        """Collapse evidence dict to a single scalar anomaly score.

        Normalizes each metric by training P95 baseline, then computes
        weighted sum.
        """
        score = 0.0
        total_weight = 0.0

        for metric_name, weight in self.weights.items():
            if metric_name not in evidence:
                continue

            raw_value = evidence[metric_name]
            if not isinstance(raw_value, (int, float)):
                continue

            # Normalize using training baseline
            if (self.training_baselines is not None
                    and metric_name in self.training_baselines):
                baseline_p95 = self.training_baselines[metric_name]['p95']
                if baseline_p95 > 1e-10:
                    normalized = raw_value / baseline_p95
                else:
                    normalized = raw_value
            else:
                normalized = raw_value

            normalized = np.clip(normalized, 0, 10)
            score += weight * normalized
            total_weight += weight

        if total_weight > 0:
            score /= total_weight

        return float(score)

    def get_evidence(self) -> Optional[List[Dict]]:
        """Get full evidence dictionaries from last detect() call."""
        return self.evidence_results
