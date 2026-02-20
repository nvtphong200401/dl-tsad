"""Pattern-Based Evidence Extraction

Detects changes in time series patterns and structure.
Metrics: ACF break, volatility spike, trend break.
"""

import numpy as np
from typing import Dict
from scipy.stats import linregress


class PatternBasedEvidence:
    """Extract pattern/structure change metrics.

    Metrics:
        1. ACF break (autocorrelation function change)
        2. Volatility spike (variance ratio)
        3. Trend break (slope/level change)
    """

    def __init__(self, acf_nlags: int = 20, volatility_rolling_window: int = 10):
        self.acf_nlags = acf_nlags
        self.volatility_rolling_window = volatility_rolling_window

    def extract(
        self,
        train_data: np.ndarray,
        test_window: np.ndarray
    ) -> Dict:
        """Extract pattern-based evidence.

        Args:
            train_data: Representative training data (M,)
            test_window: Test window values (W,)

        Returns:
            Dict with pattern change metrics.
        """
        evidence = {}
        train = train_data.flatten().astype(float)
        test = test_window.flatten().astype(float)

        if len(train) < 5 or len(test) < 5:
            return evidence

        # 1. ACF break
        self._compute_acf_break(train, test, evidence)

        # 2. Volatility spike
        self._compute_volatility(train, test, evidence)

        # 3. Trend break
        self._compute_trend_break(test, evidence)

        return evidence

    def _compute_acf(self, data: np.ndarray, nlags: int) -> np.ndarray:
        """Compute autocorrelation function.

        Tries statsmodels first, falls back to manual numpy implementation.
        """
        nlags = min(nlags, len(data) - 1)
        if nlags < 1:
            return np.array([1.0])

        try:
            from statsmodels.tsa.stattools import acf
            return acf(data, nlags=nlags, fft=True)
        except ImportError:
            pass

        # Manual implementation
        n = len(data)
        centered = data - np.mean(data)
        c0 = np.sum(centered ** 2) / n
        if c0 < 1e-10:
            return np.ones(nlags + 1)

        acf_values = [1.0]
        for k in range(1, nlags + 1):
            ck = np.sum(centered[:n - k] * centered[k:]) / n
            acf_values.append(ck / c0)
        return np.array(acf_values)

    def _compute_acf_break(
        self,
        train: np.ndarray,
        test: np.ndarray,
        evidence: Dict
    ) -> None:
        """Compare ACF between training and test data."""
        nlags = min(self.acf_nlags, len(train) - 1, len(test) - 1)
        if nlags < 2:
            evidence['max_acf_diff'] = 0.0
            evidence['mean_acf_diff'] = 0.0
            evidence['period_changed'] = False
            return

        acf_train = self._compute_acf(train, nlags)
        acf_test = self._compute_acf(test, nlags)

        # Ensure same length
        L = min(len(acf_train), len(acf_test))
        acf_diff = np.abs(acf_train[:L] - acf_test[:L])

        evidence['max_acf_diff'] = float(np.max(acf_diff))
        evidence['mean_acf_diff'] = float(np.mean(acf_diff))

        # Check if dominant period changed
        if L > 2:
            train_period = int(np.argmax(acf_train[1:L]) + 1)
            test_period = int(np.argmax(acf_test[1:L]) + 1)
            evidence['period_changed'] = train_period != test_period
        else:
            evidence['period_changed'] = False

    def _compute_volatility(
        self,
        train: np.ndarray,
        test: np.ndarray,
        evidence: Dict
    ) -> None:
        """Compute volatility ratio and rolling volatility."""
        std_train = float(np.std(train))
        std_test = float(np.std(test))
        std_safe = std_train if std_train > 1e-10 else 1e-10

        ratio = std_test / std_safe
        evidence['volatility_ratio'] = ratio
        evidence['high_volatility'] = ratio > 2.0
        evidence['extreme_volatility'] = ratio > 5.0

        # Rolling volatility within test window
        w = min(self.volatility_rolling_window, len(test) - 1)
        if w >= 2:
            rolling_std = np.array([
                np.std(test[max(0, i - w + 1):i + 1])
                for i in range(len(test))
            ])
            evidence['max_rolling_std'] = float(np.max(rolling_std))

    def _compute_trend_break(self, test: np.ndarray, evidence: Dict) -> None:
        """Detect trend breaks by comparing slopes in first/second half."""
        n = len(test)
        if n < 6:
            evidence['slope_before'] = 0.0
            evidence['slope_after'] = 0.0
            evidence['slope_diff'] = 0.0
            evidence['level_diff'] = 0.0
            evidence['trend_break'] = False
            return

        mid = n // 2
        x1 = np.arange(mid)
        x2 = np.arange(mid, n)

        try:
            slope1, intercept1, _, _, _ = linregress(x1, test[:mid])
            slope2, intercept2, _, _, _ = linregress(x2, test[mid:])
        except Exception:
            evidence['slope_diff'] = 0.0
            evidence['trend_break'] = False
            return

        slope_diff = abs(slope2 - slope1)
        level_diff = abs(intercept2 - intercept1)

        evidence['slope_before'] = float(slope1)
        evidence['slope_after'] = float(slope2)
        evidence['slope_diff'] = float(slope_diff)
        evidence['level_diff'] = float(level_diff)

        # Adaptive threshold based on data range
        data_range = np.ptp(test)
        threshold = 0.1 * data_range / max(n, 1) if data_range > 1e-10 else 0.01
        evidence['trend_break'] = slope_diff > threshold
