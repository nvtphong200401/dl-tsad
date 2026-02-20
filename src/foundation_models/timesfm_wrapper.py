"""TimesFM (Google) Foundation Model Wrapper"""

import numpy as np
from typing import Dict, Optional
from .base import FoundationModel


class TimesFMWrapper(FoundationModel):
    """Wrapper for Google's TimesFM foundation model

    TimesFM is a decoder-only transformer (200M-1.6B params) pre-trained
    on 100B+ time points from Google datasets.

    Paper: Das et al., "A decoder-only foundation model for time-series
    forecasting", ICML 2024
    """

    def __init__(self, model_name: str = "google/timesfm-1.0-200m"):
        """Initialize TimesFM wrapper

        Args:
            model_name: HuggingFace model identifier
                - "google/timesfm-1.0-200m" (default, fastest)
                - "google/timesfm-1.0-1.6b" (larger, better quality)
        """
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Load pre-trained TimesFM model"""
        try:
            import timesfm
            self.model = timesfm.TimesFM.from_pretrained(self.model_name)
            print(f"Loaded TimesFM model: {self.model_name}")
        except ImportError:
            raise ImportError(
                "TimesFM not installed. Install with: pip install timesfm"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load TimesFM model: {e}")

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        context_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate zero-shot forecast

        Args:
            context: Historical data (T,) or (T, D)
            horizon: Number of steps to forecast
            context_length: Length of context to use (default: use all)
            **kwargs: Additional TimesFM parameters

        Returns:
            Dictionary with 'forecast' and 'model' keys
        """
        if not self.is_loaded():
            self.load_model()

        # Ensure correct shape
        if context.ndim == 1:
            context = context.reshape(1, -1)  # (1, T)
        elif context.ndim == 2:
            # If multivariate (T, D), need to forecast each dimension
            # For now, average across dimensions (TODO: improve)
            if context.shape[1] > 1:
                print(f"Warning: TimesFM expects univariate. Averaging {context.shape[1]} dimensions.")
                context = context.mean(axis=1, keepdims=True).T  # (1, T)
            else:
                context = context.T  # (D, T) → transpose for batch format

        # Set context length
        if context_length is None:
            context_length = context.shape[1]

        # Generate forecast
        try:
            forecast = self.model.forecast(
                time_series=context,
                horizon=horizon,
                context_length=context_length,
                **kwargs
            )

            # Ensure output is 1D array
            if isinstance(forecast, np.ndarray):
                forecast = forecast.squeeze()
            else:
                # Convert to numpy if needed
                forecast = np.array(forecast).squeeze()

            return {
                'forecast': forecast,  # Shape: (horizon,)
                'model': 'timesfm',
                'model_name': self.model_name
            }

        except Exception as e:
            raise RuntimeError(f"TimesFM forecast failed: {e}")
