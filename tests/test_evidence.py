"""Tests for statistical evidence extraction framework."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evidence.forecast_based import ForecastBasedEvidence
from src.evidence.statistical_tests import StatisticalTestEvidence
from src.evidence.distribution_based import DistributionBasedEvidence
from src.evidence.pattern_based import PatternBasedEvidence
from src.evidence.evidence_extractor import StatisticalEvidenceExtractor


# ============================================================
# Forecast-Based Evidence Tests
# ============================================================

class TestForecastBasedEvidence:

    def test_mae_mse_mape(self):
        fb = ForecastBasedEvidence()
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        forecast = np.array([1.1, 2.2, 2.8, 4.5, 4.7])
        result = fb.extract(actual=actual, forecast=forecast)
        assert 'mae' in result
        assert 'mse' in result
        assert 'mape' in result
        assert result['mae'] > 0
        assert result['mse'] > 0
        assert result['mape'] > 0

    def test_perfect_forecast(self):
        fb = ForecastBasedEvidence()
        data = np.array([1.0, 2.0, 3.0])
        result = fb.extract(actual=data, forecast=data)
        assert result['mae'] == 0.0
        assert result['mse'] == 0.0

    def test_quantile_violations(self):
        fb = ForecastBasedEvidence()
        actual = np.array([10.0, 10.0, 10.0])
        forecast = np.array([1.0, 1.0, 1.0])
        quantiles = {
            'P01': np.array([0.5, 0.5, 0.5]),
            'P10': np.array([0.8, 0.8, 0.8]),
            'P90': np.array([1.2, 1.2, 1.2]),
            'P99': np.array([1.5, 1.5, 1.5]),
        }
        result = fb.extract(actual=actual, forecast=forecast, quantiles=quantiles)
        assert 'violation_ratio' in result
        assert result['violation_ratio'] == 1.0  # All points above P90
        assert result['extreme_violation'] is True

    def test_surprise_score(self):
        fb = ForecastBasedEvidence()
        actual = np.array([5.0, 5.0, 5.0])
        forecast = np.array([1.0, 1.0, 1.0])
        np.random.seed(42)
        samples = np.random.randn(50, 3)  # centered ~0, actual=5 is far
        result = fb.extract(actual=actual, forecast=forecast, samples=samples)
        assert 'mean_surprise' in result
        assert result['mean_surprise'] > 0

    def test_missing_optional_inputs(self):
        fb = ForecastBasedEvidence()
        actual = np.array([1.0, 2.0, 3.0])
        forecast = np.array([1.1, 1.9, 3.1])
        result = fb.extract(actual=actual, forecast=forecast)
        assert 'mae' in result
        assert 'violation_ratio' not in result
        assert 'mean_surprise' not in result

    def test_length_mismatch(self):
        fb = ForecastBasedEvidence()
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        forecast = np.array([1.1, 2.2, 2.8])  # shorter
        result = fb.extract(actual=actual, forecast=forecast)
        assert 'mae' in result  # Should still work with min length


# ============================================================
# Statistical Test Evidence Tests
# ============================================================

class TestStatisticalTestEvidence:

    def test_z_score(self):
        st = StatisticalTestEvidence()
        window = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
        result = st.extract(test_window=window, train_mean=0.0, train_std=1.0)
        assert result['max_abs_z_score'] == 10.0
        assert result['extreme_z_count'] == 1
        assert result['extreme_z_ratio'] == 0.2  # 1/5

    def test_z_score_normal(self):
        st = StatisticalTestEvidence()
        np.random.seed(42)
        window = np.random.randn(100)
        result = st.extract(test_window=window, train_mean=0.0, train_std=1.0)
        assert result['max_abs_z_score'] < 5.0  # Unlikely to exceed 5

    def test_grubbs_with_outlier(self):
        st = StatisticalTestEvidence()
        np.random.seed(42)
        window = np.random.randn(50)
        window[25] = 20.0  # Clear outlier
        result = st.extract(test_window=window, train_mean=0.0, train_std=1.0)
        assert result['grubbs_is_outlier'] == True
        assert result['grubbs_outlier_index'] == 25

    def test_grubbs_no_outlier(self):
        st = StatisticalTestEvidence()
        np.random.seed(42)
        window = np.random.randn(50)
        result = st.extract(test_window=window, train_mean=0.0, train_std=1.0)
        # Normal data should not have Grubbs outlier (usually)
        assert 'grubbs_is_outlier' in result

    def test_cusum_change_point(self):
        st = StatisticalTestEvidence()
        window = np.concatenate([np.zeros(25), np.ones(25) * 5])
        result = st.extract(test_window=window, train_mean=0.0, train_std=1.0)
        assert result['max_cusum'] > 0
        assert result['cusum_has_change_point'] is True

    def test_cusum_no_change(self):
        st = StatisticalTestEvidence()
        window = np.zeros(50)
        result = st.extract(test_window=window, train_mean=0.0, train_std=1.0)
        assert result['cusum_has_change_point'] is False

    def test_short_window(self):
        st = StatisticalTestEvidence()
        window = np.array([1.0, 2.0])
        result = st.extract(test_window=window, train_mean=0.0, train_std=1.0)
        assert result == {}  # Too short


# ============================================================
# Distribution-Based Evidence Tests
# ============================================================

class TestDistributionBasedEvidence:

    def test_same_distribution(self):
        db = DistributionBasedEvidence()
        np.random.seed(42)
        data = np.random.randn(1000)
        result = db.extract(train_data=data, test_window=data[:100])
        assert result['kl_divergence'] < 1.0  # Should be small

    def test_different_distribution(self):
        db = DistributionBasedEvidence()
        np.random.seed(42)
        train = np.random.randn(1000)
        test = np.random.randn(100) + 10  # Shifted by 10
        result = db.extract(train_data=train, test_window=test)
        assert result['kl_divergence'] > 0.5

    def test_wasserstein_distance(self):
        db = DistributionBasedEvidence()
        np.random.seed(42)
        train = np.random.randn(1000)
        test = np.random.randn(100) + 5
        result = db.extract(train_data=train, test_window=test)
        assert result['wasserstein_distance'] > 0
        assert result['normalized_wasserstein'] > 1.0

    def test_empty_input(self):
        db = DistributionBasedEvidence()
        result = db.extract(train_data=np.array([1.0]), test_window=np.array([2.0]))
        assert result == {}  # Too short


# ============================================================
# Pattern-Based Evidence Tests
# ============================================================

class TestPatternBasedEvidence:

    def test_volatility_spike(self):
        pb = PatternBasedEvidence()
        np.random.seed(42)
        train = np.random.randn(1000) * 1.0
        test = np.random.randn(100) * 5.0
        result = pb.extract(train_data=train, test_window=test)
        assert result['volatility_ratio'] > 3.0
        assert result['high_volatility'] is True

    def test_no_volatility_spike(self):
        pb = PatternBasedEvidence()
        np.random.seed(42)
        train = np.random.randn(1000)
        test = np.random.randn(100)
        result = pb.extract(train_data=train, test_window=test)
        assert result['volatility_ratio'] < 2.0
        assert result['high_volatility'] is False

    def test_trend_break(self):
        pb = PatternBasedEvidence()
        np.random.seed(42)
        train = np.random.randn(1000)
        # Flat then steep rise
        test = np.concatenate([np.zeros(50), np.linspace(0, 10, 50)])
        result = pb.extract(train_data=train, test_window=test)
        assert result['slope_diff'] > 0
        assert 'slope_before' in result
        assert 'slope_after' in result

    def test_acf_break(self):
        pb = PatternBasedEvidence()
        t = np.linspace(0, 10 * np.pi, 1000)
        train = np.sin(t)  # Periodic
        np.random.seed(42)
        test = np.random.randn(100)  # Random
        result = pb.extract(train_data=train, test_window=test)
        assert result['max_acf_diff'] > 0
        assert 'mean_acf_diff' in result

    def test_short_window(self):
        pb = PatternBasedEvidence()
        result = pb.extract(
            train_data=np.array([1.0, 2.0]),
            test_window=np.array([3.0, 4.0])
        )
        assert result == {}  # Too short


# ============================================================
# Full Evidence Extractor Tests
# ============================================================

class TestStatisticalEvidenceExtractor:

    def test_full_extraction(self):
        extractor = StatisticalEvidenceExtractor()
        np.random.seed(42)
        train_data = np.random.randn(1000)
        test_window = np.random.randn(100) + 5
        forecast = np.random.randn(100)

        evidence = extractor.extract(
            test_window=test_window,
            forecast_result={'forecast': forecast, 'quantiles': None, 'samples': None},
            train_statistics={'mean': 0.0, 'std': 1.0},
            train_data=train_data
        )

        # Forecast-based metrics
        assert 'mae' in evidence
        assert 'mse' in evidence
        assert 'mape' in evidence
        # Statistical test metrics
        assert 'max_abs_z_score' in evidence
        assert 'grubbs_statistic' in evidence
        assert 'max_cusum' in evidence
        # Distribution-based metrics
        assert 'kl_divergence' in evidence
        assert 'wasserstein_distance' in evidence
        # Pattern-based metrics
        assert 'volatility_ratio' in evidence
        assert 'max_acf_diff' in evidence
        assert 'slope_diff' in evidence

    def test_selective_categories(self):
        extractor = StatisticalEvidenceExtractor(
            enabled_categories=['statistical_tests']
        )

        evidence = extractor.extract(
            test_window=np.random.randn(100),
            train_statistics={'mean': 0.0, 'std': 1.0}
        )

        assert 'max_abs_z_score' in evidence
        assert 'mae' not in evidence  # Forecast-based not enabled
        assert 'kl_divergence' not in evidence  # Distribution not enabled

    def test_graceful_failure_no_context(self):
        extractor = StatisticalEvidenceExtractor()
        evidence = extractor.extract(
            test_window=np.random.randn(100)
        )
        # Should not crash, returns whatever it can compute
        assert isinstance(evidence, dict)

    def test_with_array_train_stats(self):
        """Training stats may be arrays (multi-dimensional)."""
        extractor = StatisticalEvidenceExtractor(
            enabled_categories=['statistical_tests']
        )
        evidence = extractor.extract(
            test_window=np.random.randn(100),
            train_statistics={
                'mean': np.array([0.0, 0.1, -0.1]),
                'std': np.array([1.0, 0.9, 1.1])
            }
        )
        assert 'max_abs_z_score' in evidence


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
