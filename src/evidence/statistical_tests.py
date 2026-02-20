"""Statistical Test Evidence Extraction

Classical statistical tests for outliers and change points.
Metrics: Z-score, Grubbs test, CUSUM.
"""

import numpy as np
from typing import Dict
from scipy.stats import t as t_dist


class StatisticalTestEvidence:
    """Extract statistical test evidence.

    Metrics:
        1. Z-score (standard deviations from training mean)
        2. Grubbs test (outlier detection within window)
        3. CUSUM (cumulative sum change-point detection)
    """

    def extract(
        self,
        test_window: np.ndarray,
        train_mean: float,
        train_std: float
    ) -> Dict:
        """Extract statistical test evidence.

        Args:
            test_window: Test window values (W,)
            train_mean: Training data mean (scalar)
            train_std: Training data std (scalar)

        Returns:
            Dict with statistical test metrics.
        """
        evidence = {}
        window = test_window.flatten().astype(float)

        if len(window) < 3:
            return evidence

        # 1. Z-score
        self._compute_z_score(window, train_mean, train_std, evidence)

        # 2. Grubbs test
        self._compute_grubbs(window, evidence)

        # 3. CUSUM
        self._compute_cusum(window, train_mean, train_std, evidence)

        return evidence

    def _compute_z_score(
        self,
        window: np.ndarray,
        train_mean: float,
        train_std: float,
        evidence: Dict
    ) -> None:
        """Compute z-scores relative to training distribution."""
        std_safe = train_std if train_std > 1e-10 else 1e-10
        z_scores = (window - train_mean) / std_safe

        abs_z = np.abs(z_scores)
        evidence['max_abs_z_score'] = float(np.max(abs_z))
        evidence['mean_abs_z_score'] = float(np.mean(abs_z))
        evidence['extreme_z_count'] = int(np.sum(abs_z > 3))
        evidence['extreme_z_ratio'] = float(np.mean(abs_z > 3))

    def _compute_grubbs(self, window: np.ndarray, evidence: Dict) -> None:
        """Compute Grubbs test for outlier detection within the window."""
        n = len(window)
        if n < 3:
            evidence['grubbs_statistic'] = 0.0
            evidence['grubbs_is_outlier'] = False
            evidence['grubbs_outlier_index'] = -1
            return

        mean_w = np.mean(window)
        std_w = np.std(window, ddof=1)

        if std_w < 1e-10:
            evidence['grubbs_statistic'] = 0.0
            evidence['grubbs_is_outlier'] = False
            evidence['grubbs_outlier_index'] = -1
            return

        # Grubbs statistic
        deviations = np.abs(window - mean_w)
        G = float(np.max(deviations) / std_w)

        # Critical value (alpha=0.05)
        alpha = 0.05
        t_crit = t_dist.ppf(1 - alpha / (2 * n), n - 2)
        G_crit = ((n - 1) * np.sqrt(t_crit ** 2)) / np.sqrt(n * (n - 2 + t_crit ** 2))

        evidence['grubbs_statistic'] = G
        evidence['grubbs_critical'] = float(G_crit)
        evidence['grubbs_is_outlier'] = G > G_crit
        evidence['grubbs_outlier_index'] = int(np.argmax(deviations))

    def _compute_cusum(
        self,
        window: np.ndarray,
        train_mean: float,
        train_std: float,
        evidence: Dict
    ) -> None:
        """Compute CUSUM for change-point detection."""
        std_safe = train_std if train_std > 1e-10 else 1e-10
        slack = 0.5 * std_safe
        threshold = 5.0 * std_safe

        n = len(window)
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)

        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + (window[i] - train_mean) - slack)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - (window[i] - train_mean) - slack)

        max_cusum = max(float(np.max(cusum_pos)), float(np.max(cusum_neg)))
        change_points = (cusum_pos > threshold) | (cusum_neg > threshold)

        evidence['max_cusum'] = max_cusum
        evidence['cusum_has_change_point'] = bool(np.any(change_points))
        evidence['cusum_change_point_count'] = int(np.sum(change_points))
