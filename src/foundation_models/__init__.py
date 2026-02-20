"""Foundation Models for Zero-Shot Time Series Forecasting

This module provides wrappers for pre-trained foundation models:
- TimesFM (Google): Deterministic forecasting
- Chronos (Amazon): Probabilistic forecasting with quantiles
- Ensemble: Combines multiple models
"""

from .base import FoundationModel
from .timesfm_wrapper import TimesFMWrapper
from .chronos_wrapper import ChronosWrapper
from .ensemble import EnsembleForecaster

__all__ = [
    'FoundationModel',
    'TimesFMWrapper',
    'ChronosWrapper',
    'EnsembleForecaster'
]
