"""TimesFM (Google) Foundation Model Wrapper

Supports TimesFM 2.0/2.5 with PyTorch backend.
Install: pip install timesfm[torch]  (or clone from github for 2.5)
"""

import numpy as np
from typing import Dict, Optional
from .base import FoundationModel


class TimesFMWrapper(FoundationModel):
    """Wrapper for Google's TimesFM foundation model

    TimesFM is a decoder-only transformer pre-trained on 100B+ time points.
    Returns both point forecasts and quantile predictions.

    Paper: Das et al., "A decoder-only foundation model for time-series
    forecasting", ICML 2024
    """

    def __init__(self, model_name: str = "google/timesfm-2.5-200m-pytorch"):
        """Initialize TimesFM wrapper

        Args:
            model_name: HuggingFace model identifier
                - "google/timesfm-2.5-200m-pytorch" (default, install from github)
        """
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Load pre-trained TimesFM model with PyTorch backend"""
        try:
            import timesfm

            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.model_name
            )
            print(f"Loaded TimesFM model: {self.model_name}")
        except ImportError:
            raise ImportError(
                "TimesFM not installed. Install with: "
                "pip install git+https://github.com/google-research/timesfm.git"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load TimesFM model: {e}")

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate zero-shot forecast with quantiles

        Args:
            context: Historical data (T,) or (T, D)
            horizon: Number of steps to forecast
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
                - 'forecast': Point forecast (H,)
                - 'quantiles': Dict of quantile arrays
                - 'model': 'timesfm'
        """
        if not self.is_loaded():
            self.load_model()

        # Ensure 1D input
        if context.ndim == 2:
            if context.shape[1] > 1:
                context = context.mean(axis=1)
            else:
                context = context.squeeze()

        # TimesFM accepts list of arrays
        try:
            point_forecast, quantile_forecast = self.model.forecast(
                horizon=horizon,
                inputs=[context.astype(np.float32)],
            )

            # point_forecast: (1, H), quantile_forecast: (1, H, Q)
            point = point_forecast[0]  # (H,)

            # Build quantiles dict from quantile_forecast
            quantiles = {}
            if quantile_forecast is not None and len(quantile_forecast.shape) == 3:
                q_data = quantile_forecast[0]  # (H, Q)
                n_quantiles = q_data.shape[1]
                # TimesFM returns mean + 10th to 90th percentiles
                if n_quantiles >= 5:
                    quantiles['P10'] = q_data[:, 1] if n_quantiles > 1 else point
                    quantiles['P50'] = point
                    quantiles['P90'] = q_data[:, -2] if n_quantiles > 1 else point

            return {
                'forecast': point,
                'quantiles': quantiles if quantiles else None,
                'model': 'timesfm',
                'model_name': self.model_name,
                'uncertainty': (quantiles.get('P90', point) - quantiles.get('P10', point))
                               if quantiles else np.zeros_like(point)
            }

        except Exception as e:
            raise RuntimeError(f"TimesFM forecast failed: {e}")

    def forecast_batch(
        self,
        contexts: list,
        horizon: int,
        **kwargs
    ) -> list:
        """Batch forecast — TimesFM natively supports list of arrays."""
        if not self.is_loaded():
            self.load_model()

        inputs = [c.astype(np.float32).flatten() for c in contexts]
        point_forecasts, quantile_forecasts = self.model.forecast(
            horizon=horizon,
            inputs=inputs,
        )
        # point_forecasts: (N, H), quantile_forecasts: (N, H, Q)
        results = []
        for i in range(len(inputs)):
            point = point_forecasts[i]
            quantiles = {}
            if quantile_forecasts is not None and len(quantile_forecasts.shape) == 3:
                q = quantile_forecasts[i]
                if q.shape[1] >= 5:
                    quantiles['P10'] = q[:, 1]
                    quantiles['P50'] = point
                    quantiles['P90'] = q[:, -2]
            results.append({
                'forecast': point,
                'quantiles': quantiles if quantiles else None,
                'model': 'timesfm',
            })
        return results
