"""Foundation Model-based Data Processor for Step 1

This processor enhances the baseline windowing + normalization with
foundation model forecasting (TimesFM, Chronos).
"""

import numpy as np
from typing import Dict, Optional, List
from .step1_data_processing import DataProcessor, WindowConfig
import sys
from pathlib import Path

# Add src to path for foundation_models import
sys.path.insert(0, str(Path(__file__).parent.parent))
from foundation_models import EnsembleForecaster


class FoundationModelProcessor(DataProcessor):
    """Enhanced data processor with foundation model forecasting

    Extends Phase 1 DataProcessor to:
    1. Create sliding windows (inherited)
    2. Normalize windows (inherited from RawWindowProcessor)
    3. Generate forecasts for each window using foundation models
    4. Store forecasts as attributes for use in Step 2

    **Backward Compatible**: Returns np.ndarray like Phase 1 processors,
    but stores additional forecast data as attributes.
    """

    def __init__(
        self,
        window_config: WindowConfig,
        forecast_horizon: Optional[int] = None,
        models: List[str] = ['chronos'],
        timesfm_model: str = "google/timesfm-1.0-200m",
        chronos_model: str = "amazon/chronos-t5-tiny",  # Use tiny for speed
        ensemble_strategy: str = 'average',
        num_samples: int = 50  # Reduced for speed
    ):
        """Initialize foundation model processor

        Args:
            window_config: Window configuration (size, stride, padding)
            forecast_horizon: Number of steps to forecast (default: same as window_size)
            models: List of foundation models to use ['timesfm', 'chronos']
            timesfm_model: TimesFM model name
            chronos_model: Chronos model name
            ensemble_strategy: How to combine models ('average', 'chronos_only', etc.)
            num_samples: Number of samples for Chronos quantile estimation
        """
        super().__init__(window_config)

        self.forecast_horizon = forecast_horizon or window_config.window_size
        self.ensemble_strategy = ensemble_strategy
        self.num_samples = num_samples

        # Initialize foundation model ensemble
        self.forecaster = EnsembleForecaster(
            models=models,
            timesfm_model=timesfm_model,
            chronos_model=chronos_model
        )

        # Storage for training statistics and forecasts
        self.train_mean = None
        self.train_std = None
        self.train_statistics = None
        self.forecast_results = None  # Store forecasts for Step 2

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """Fit on training windows and normalize

        Args:
            windows: Windowed training data (N, W, D)

        Returns:
            Normalized windows (N, W, D) - backward compatible with Phase 1
        """
        N, W, D = windows.shape

        # Compute training statistics (for Step 2 evidence extraction)
        windows_flat = windows.reshape(-1, D)
        self.train_mean = np.mean(windows_flat, axis=0)
        self.train_std = np.std(windows_flat, axis=0) + 1e-10  # Avoid division by zero

        # Store comprehensive statistics
        self.train_statistics = {
            'mean': self.train_mean,
            'std': self.train_std,
            'quantiles': {
                f'P{int(q*100)}': np.quantile(windows_flat, q, axis=0)
                for q in [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
            },
            'min': np.min(windows_flat, axis=0),
            'max': np.max(windows_flat, axis=0)
        }

        # Normalize windows
        normalized = (windows_flat - self.train_mean) / self.train_std
        normalized = normalized.reshape(N, W, D)

        print(f"  Foundation model processor fitted on {N} windows")
        print(f"  Training statistics: mean={self.train_mean.mean():.2f}, std={self.train_std.mean():.2f}")

        return normalized

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Transform test windows and generate forecasts

        Args:
            windows: Windowed test data (N, W, D)

        Returns:
            Normalized windows (N, W, D) - backward compatible with Phase 1

        Side Effect:
            Stores forecasts in self.forecast_results for Step 2
        """
        if self.train_mean is None:
            raise ValueError("Must call fit_transform() before transform()")

        N, W, D = windows.shape

        # Normalize using training statistics
        windows_flat = windows.reshape(-1, D)
        normalized = (windows_flat - self.train_mean) / self.train_std
        normalized = normalized.reshape(N, W, D)

        # Generate forecasts for each window
        print(f"  Generating forecasts for {N} windows...")
        self.forecast_results = []

        for i, window in enumerate(normalized):
            # Use window as context for forecasting
            # For multivariate, use first dimension (TODO: improve)
            context = window[:, 0] if D > 1 else window.squeeze()

            try:
                forecast_result = self.forecaster.forecast(
                    context=context,
                    horizon=self.forecast_horizon,
                    strategy=self.ensemble_strategy,
                    num_samples=self.num_samples
                )
                self.forecast_results.append(forecast_result)

            except Exception as e:
                print(f"    Warning: Forecast failed for window {i}: {e}")
                # Create dummy forecast
                self.forecast_results.append({
                    'forecast': np.zeros(self.forecast_horizon),
                    'quantiles': None,
                    'error': str(e)
                })

        print(f"  Generated {len(self.forecast_results)} forecasts")

        return normalized

    def get_output_dim(self) -> int:
        """Return output dimension (same as input after normalization)"""
        return 1 if self.train_mean is None else len(self.train_mean)

    def get_forecasts(self) -> Optional[List[Dict]]:
        """Get stored forecast results

        Returns:
            List of forecast dictionaries from transform()
            None if transform() hasn't been called yet
        """
        return self.forecast_results

    def get_train_statistics(self) -> Optional[Dict]:
        """Get training statistics

        Returns:
            Dictionary with mean, std, quantiles, etc.
            None if fit_transform() hasn't been called yet
        """
        return self.train_statistics
