"""Statistical Evidence Extractor

Main orchestrator that combines all 4 evidence categories (13 metrics total):
- Forecast-Based (5): MAE, MSE, MAPE, quantile violations, surprise
- Statistical Tests (3): Z-score, Grubbs, CUSUM
- Distribution-Based (2): KL divergence, Wasserstein distance
- Pattern-Based (3): ACF break, volatility spike, trend break
"""

import numpy as np
from typing import Dict, List, Optional

from .forecast_based import ForecastBasedEvidence
from .statistical_tests import StatisticalTestEvidence
from .distribution_based import DistributionBasedEvidence
from .pattern_based import PatternBasedEvidence


class StatisticalEvidenceExtractor:
    """Extract comprehensive statistical evidence for anomaly detection.

    Args:
        enabled_categories: List of category names to enable.
            Options: 'forecast_based', 'statistical_tests',
            'distribution_based', 'pattern_based'
        config: Optional per-category configuration dict.
    """

    def __init__(
        self,
        enabled_categories: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        self.config = config or {}
        self.enabled_categories = enabled_categories or [
            'forecast_based',
            'statistical_tests',
            'distribution_based',
            'pattern_based'
        ]

        # Initialize enabled extractors
        self.extractors = {}
        if 'forecast_based' in self.enabled_categories:
            self.extractors['forecast_based'] = ForecastBasedEvidence()
        if 'statistical_tests' in self.enabled_categories:
            self.extractors['statistical_tests'] = StatisticalTestEvidence()
        if 'distribution_based' in self.enabled_categories:
            dist_cfg = self.config.get('distribution_based', {})
            self.extractors['distribution_based'] = DistributionBasedEvidence(
                n_bins=dist_cfg.get('n_bins', 20)
            )
        if 'pattern_based' in self.enabled_categories:
            pat_cfg = self.config.get('pattern_based', {})
            self.extractors['pattern_based'] = PatternBasedEvidence(
                acf_nlags=pat_cfg.get('acf_nlags', 20),
                volatility_rolling_window=pat_cfg.get('volatility_rolling_window', 10)
            )

    def extract(
        self,
        test_window: np.ndarray,
        forecast_result: Optional[Dict] = None,
        train_statistics: Optional[Dict] = None,
        train_data: Optional[np.ndarray] = None
    ) -> Dict:
        """Extract all enabled evidence metrics for a single window.

        Args:
            test_window: Test window to analyze (W,) or (W, D)
            forecast_result: Foundation model forecast dict with keys:
                'forecast', 'quantiles', 'samples', 'uncertainty'
            train_statistics: Training stats dict with keys:
                'mean', 'std', 'quantiles', 'min', 'max'
            train_data: Representative training data for distribution
                comparison (M,)

        Returns:
            Dict with all computed evidence metrics.
        """
        evidence = {}

        # Ensure 1D
        window_1d = test_window.squeeze() if test_window.ndim > 1 else test_window

        # Category 1: Forecast-based
        if 'forecast_based' in self.extractors and forecast_result is not None:
            forecast = forecast_result.get('forecast')
            if forecast is not None:
                try:
                    fb = self.extractors['forecast_based'].extract(
                        actual=window_1d,
                        forecast=forecast,
                        quantiles=forecast_result.get('quantiles'),
                        samples=forecast_result.get('samples')
                    )
                    evidence.update(fb)
                except Exception as e:
                    evidence['forecast_based_error'] = str(e)

        # Category 2: Statistical tests
        if 'statistical_tests' in self.extractors and train_statistics is not None:
            try:
                train_mean = train_statistics['mean']
                train_std = train_statistics['std']
                # Handle array values (multi-dim)
                if isinstance(train_mean, np.ndarray):
                    train_mean = float(train_mean.mean())
                if isinstance(train_std, np.ndarray):
                    train_std = float(train_std.mean())

                st = self.extractors['statistical_tests'].extract(
                    test_window=window_1d,
                    train_mean=train_mean,
                    train_std=train_std
                )
                evidence.update(st)
            except Exception as e:
                evidence['statistical_tests_error'] = str(e)

        # Category 3: Distribution-based
        if 'distribution_based' in self.extractors and train_data is not None:
            try:
                db = self.extractors['distribution_based'].extract(
                    train_data=train_data,
                    test_window=window_1d
                )
                evidence.update(db)
            except Exception as e:
                evidence['distribution_based_error'] = str(e)

        # Category 4: Pattern-based
        if 'pattern_based' in self.extractors and train_data is not None:
            try:
                pb = self.extractors['pattern_based'].extract(
                    train_data=train_data,
                    test_window=window_1d
                )
                evidence.update(pb)
            except Exception as e:
                evidence['pattern_based_error'] = str(e)

        return evidence
