"""Statistical Evidence Framework for Anomaly Detection

Extracts 13 independent statistical metrics across 4 categories:
- Forecast-Based: MAE, MSE, MAPE, quantile violations, surprise score
- Statistical Tests: Z-score, Grubbs test, CUSUM
- Distribution-Based: KL divergence, Wasserstein distance
- Pattern-Based: ACF break, volatility spike, trend break
"""

from .evidence_extractor import StatisticalEvidenceExtractor
from .forecast_based import ForecastBasedEvidence
from .statistical_tests import StatisticalTestEvidence
from .distribution_based import DistributionBasedEvidence
from .pattern_based import PatternBasedEvidence

__all__ = [
    'StatisticalEvidenceExtractor',
    'ForecastBasedEvidence',
    'StatisticalTestEvidence',
    'DistributionBasedEvidence',
    'PatternBasedEvidence',
]
