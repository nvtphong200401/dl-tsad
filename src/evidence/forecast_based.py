"""Forecast-Based Evidence Extraction

Compares actual window values against foundation model predictions.
Metrics: MAE, MSE, MAPE, quantile violations, surprise score.
"""

import numpy as np
from typing import Dict, Optional


class ForecastBasedEvidence:
    """Extract forecast error metrics.

    Metrics:
        1. MAE (Mean Absolute Error)
        2. MSE (Mean Squared Error)
        3. MAPE (Mean Absolute Percentage Error)
        4. Quantile violations (requires Chronos quantiles)
        5. Surprise score (requires Chronos samples)
    """

    def extract(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
        quantiles: Optional[Dict[str, np.ndarray]] = None,
        samples: Optional[np.ndarray] = None
    ) -> Dict:
        """Extract forecast-based evidence.

        Args:
            actual: Actual window values (W,)
            forecast: Point forecast (H,)
            quantiles: Optional dict {'P01': ..., 'P10': ..., 'P90': ..., 'P99': ...}
            samples: Optional forecast samples (num_samples, H)

        Returns:
            Dict with forecast error metrics.
        """
        evidence = {}

        # Align lengths
        L = min(len(actual), len(forecast))
        if L == 0:
            return evidence
        a = actual[:L].astype(float)
        f = forecast[:L].astype(float)

        # 1. MAE
        errors = np.abs(a - f)
        evidence['mae'] = float(np.mean(errors))

        # 2. MSE
        evidence['mse'] = float(np.mean((a - f) ** 2))

        # 3. MAPE
        eps = 1e-10
        evidence['mape'] = float(np.mean(np.abs((a - f) / (np.abs(a) + eps))) * 100)

        # 4. Quantile violations
        if quantiles is not None:
            self._compute_quantile_violations(a, quantiles, L, evidence)

        # 5. Surprise score
        if samples is not None and samples.shape[1] >= L:
            self._compute_surprise(a, samples[:, :L], evidence)

        return evidence

    def _compute_quantile_violations(
        self,
        actual: np.ndarray,
        quantiles: Dict[str, np.ndarray],
        L: int,
        evidence: Dict
    ) -> None:
        """Compute fraction of points violating confidence bands."""
        n_violations = 0
        n_extreme = 0
        n_checked = 0

        p10 = quantiles.get('P10')
        p90 = quantiles.get('P90')
        p01 = quantiles.get('P01')
        p99 = quantiles.get('P99')

        for i in range(L):
            n_checked += 1
            # Standard violation (outside 80% CI)
            if p10 is not None and i < len(p10) and actual[i] < p10[i]:
                n_violations += 1
            elif p90 is not None and i < len(p90) and actual[i] > p90[i]:
                n_violations += 1

            # Extreme violation (outside 98% CI)
            if p01 is not None and i < len(p01) and actual[i] < p01[i]:
                n_extreme += 1
            elif p99 is not None and i < len(p99) and actual[i] > p99[i]:
                n_extreme += 1

        if n_checked > 0:
            evidence['violation_ratio'] = float(n_violations / n_checked)
            evidence['extreme_violation_ratio'] = float(n_extreme / n_checked)
            evidence['extreme_violation'] = n_extreme > 0

    def _compute_surprise(
        self,
        actual: np.ndarray,
        samples: np.ndarray,
        evidence: Dict
    ) -> None:
        """Compute surprise score using percentile rank (fast mode).

        Uses empirical CDF rank instead of KDE for speed.
        """
        surprise_scores = []
        for i in range(len(actual)):
            sample_col = samples[:, i]
            if np.std(sample_col) < 1e-10:
                surprise_scores.append(0.0)
                continue

            # Percentile rank: how extreme is actual relative to samples?
            rank = np.mean(sample_col <= actual[i])
            # Convert to surprise: points near 0 or 1 are surprising
            deviation = abs(rank - 0.5) * 2  # 0=expected, 1=extreme
            surprise = -np.log(1 - deviation + 1e-10)
            surprise_scores.append(float(surprise))

        if surprise_scores:
            evidence['mean_surprise'] = float(np.mean(surprise_scores))
            evidence['max_surprise'] = float(np.max(surprise_scores))
