"""Chronos (Amazon) Foundation Model Wrapper"""

import numpy as np
from typing import Dict, Optional
from .base import FoundationModel


class ChronosWrapper(FoundationModel):
    """Wrapper for Amazon's Chronos foundation model

    Chronos is a T5-based probabilistic forecasting model that provides
    quantile predictions for uncertainty quantification.

    Paper: Ansari et al., "Chronos: Learning the Language of Time Series", 2024
    """

    def __init__(self, model_name: str = "amazon/chronos-t5-small"):
        """Initialize Chronos wrapper

        Args:
            model_name: HuggingFace model identifier
                - "amazon/chronos-t5-tiny" (fastest, least accurate)
                - "amazon/chronos-t5-mini" (fast, good)
                - "amazon/chronos-t5-small" (default, balanced)
                - "amazon/chronos-t5-base" (slower, better)
                - "amazon/chronos-t5-large" (slowest, best)
        """
        self.model_name = model_name
        self.pipeline = None

    def load_model(self):
        """Load pre-trained Chronos model"""
        try:
            import torch
            from chronos import ChronosPipeline
            use_cuda = torch.cuda.is_available()

            # Try multiple loading strategies (different envs need different approaches)
            pipeline = None
            device_used = "cpu"

            if use_cuda:
                # Strategy 1: device_map="cuda" (works on most setups)
                try:
                    pipeline = ChronosPipeline.from_pretrained(
                        self.model_name,
                        device_map="cuda",
                        torch_dtype=torch.float32,
                    )
                    device_used = "cuda"
                except Exception:
                    pass

                # Strategy 2: Load on CPU, move inner model to GPU
                if pipeline is None:
                    try:
                        pipeline = ChronosPipeline.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float32,
                        )
                        if hasattr(pipeline, 'model'):
                            pipeline.model = pipeline.model.to("cuda")
                        if hasattr(pipeline, 'device'):
                            pipeline.device = torch.device("cuda")
                        device_used = "cuda"
                    except Exception:
                        pipeline = None

            # Strategy 3: CPU fallback (always works)
            if pipeline is None:
                pipeline = ChronosPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                )
                device_used = "cpu"

            self.pipeline = pipeline
            self.device = torch.device(device_used)
            print(f"Loaded Chronos model: {self.model_name} on {device_used}")
        except ImportError:
            raise ImportError(
                "Chronos not installed. Install with: pip install chronos-forecasting"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Chronos model: {e}")

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        num_samples: int = 100,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate probabilistic forecast with quantiles

        Args:
            context: Historical data (T,) or (T, D)
            horizon: Number of steps to forecast
            num_samples: Number of samples for quantile estimation (default: 100)
            **kwargs: Additional Chronos parameters

        Returns:
            Dictionary with:
                - 'forecast': Median forecast (H,)
                - 'quantiles': Dict of quantile arrays {P01, P10, P50, P90, P99}
                - 'samples': All forecast samples (num_samples, H)
                - 'model': 'chronos'
        """
        if not self.is_loaded():
            self.load_model()

        # Ensure correct shape for Chronos
        if context.ndim == 1:
            context = context.reshape(1, -1)  # (1, T)
        elif context.ndim == 2:
            # If multivariate (T, D), average dimensions (TODO: improve)
            if context.shape[1] > 1:
                print(f"Warning: Chronos expects univariate. Averaging {context.shape[1]} dimensions.")
                context = context.mean(axis=1, keepdims=True).T  # (1, T)
            else:
                context = context.T  # (D, T) → (1, T)

        # Generate forecast samples
        try:
            # Chronos returns tensor, convert to numpy
            import torch

            samples = self.pipeline.predict(
                torch.tensor(context),
                prediction_length=horizon,
                num_samples=num_samples,
                **kwargs
            )

            # Convert to numpy: (batch, num_samples, horizon) → (num_samples, horizon)
            samples_np = samples.cpu().numpy().squeeze(0)  # (num_samples, horizon)

            # Compute quantiles
            quantiles = {
                'P01': np.quantile(samples_np, 0.01, axis=0),  # (horizon,)
                'P10': np.quantile(samples_np, 0.10, axis=0),
                'P25': np.quantile(samples_np, 0.25, axis=0),
                'P50': np.quantile(samples_np, 0.50, axis=0),  # Median
                'P75': np.quantile(samples_np, 0.75, axis=0),
                'P90': np.quantile(samples_np, 0.90, axis=0),
                'P99': np.quantile(samples_np, 0.99, axis=0),
            }

            return {
                'forecast': quantiles['P50'],  # Use median as point forecast
                'quantiles': quantiles,
                'samples': samples_np,
                'model': 'chronos',
                'model_name': self.model_name,
                'uncertainty': quantiles['P90'] - quantiles['P10']  # IQR as uncertainty
            }

        except Exception as e:
            raise RuntimeError(f"Chronos forecast failed: {e}")

    def get_confidence_interval(
        self,
        forecast_result: Dict,
        confidence: float = 0.90
    ) -> tuple:
        """Extract confidence interval from forecast

        Args:
            forecast_result: Output from forecast()
            confidence: Confidence level (0.90 = 90% CI)

        Returns:
            (lower_bound, upper_bound) arrays
        """
        if confidence == 0.90:
            return (forecast_result['quantiles']['P10'],
                    forecast_result['quantiles']['P90'])
        elif confidence == 0.80:
            return (forecast_result['quantiles']['P10'],
                    forecast_result['quantiles']['P90'])
        elif confidence == 0.98:
            return (forecast_result['quantiles']['P01'],
                    forecast_result['quantiles']['P99'])
        else:
            raise ValueError(f"Confidence {confidence} not supported. Use 0.80, 0.90, or 0.98")

    def forecast_batch(
        self,
        contexts: list,
        horizon: int,
        num_samples: int = 50,
        **kwargs
    ) -> list:
        """Batch forecast — Chronos supports batched tensor input."""
        if not self.is_loaded():
            self.load_model()

        import torch

        # Stack all contexts into a single batch tensor (N, T)
        batch = np.stack([c.flatten() for c in contexts])  # (N, T)
        batch_tensor = torch.tensor(batch, dtype=torch.float32)

        samples = self.pipeline.predict(
            batch_tensor,
            prediction_length=horizon,
            num_samples=num_samples,
        )
        # samples: (N, num_samples, H)
        samples_np = samples.cpu().numpy()

        results = []
        for i in range(len(contexts)):
            s = samples_np[i]  # (num_samples, H)
            quantiles = {
                'P01': np.quantile(s, 0.01, axis=0),
                'P10': np.quantile(s, 0.10, axis=0),
                'P25': np.quantile(s, 0.25, axis=0),
                'P50': np.quantile(s, 0.50, axis=0),
                'P75': np.quantile(s, 0.75, axis=0),
                'P90': np.quantile(s, 0.90, axis=0),
                'P99': np.quantile(s, 0.99, axis=0),
            }
            results.append({
                'forecast': quantiles['P50'],
                'quantiles': quantiles,
                'samples': s,
                'model': 'chronos',
            })
        return results
