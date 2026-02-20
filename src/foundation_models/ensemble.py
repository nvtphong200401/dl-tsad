"""Ensemble Forecaster combining TimesFM and Chronos"""

import numpy as np
from typing import Dict, List, Optional
from .timesfm_wrapper import TimesFMWrapper
from .chronos_wrapper import ChronosWrapper


class EnsembleForecaster:
    """Ensemble forecasting combining multiple foundation models

    Combines TimesFM (deterministic) and Chronos (probabilistic) for
    robust forecasts with uncertainty quantification.
    """

    def __init__(
        self,
        models: List[str] = ['chronos'],  # Default to Chronos only (more reliable)
        timesfm_model: str = "google/timesfm-1.0-200m",
        chronos_model: str = "amazon/chronos-t5-small"
    ):
        """Initialize ensemble forecaster

        Args:
            models: List of models to use ['timesfm', 'chronos']
            timesfm_model: TimesFM model name
            chronos_model: Chronos model name
        """
        self.model_names = models
        self.models = {}

        # Initialize models
        if 'timesfm' in models:
            self.models['timesfm'] = TimesFMWrapper(model_name=timesfm_model)

        if 'chronos' in models:
            self.models['chronos'] = ChronosWrapper(model_name=chronos_model)

        if not self.models:
            raise ValueError("Must specify at least one model")

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        strategy: str = 'average',
        num_samples: int = 100,
        **kwargs
    ) -> Dict:
        """Generate ensemble forecast

        Args:
            context: Historical data (T,) or (T, D)
            horizon: Number of steps to forecast
            strategy: Ensemble strategy
                - 'average': Simple average of forecasts
                - 'chronos_only': Use only Chronos
                - 'timesfm_only': Use only TimesFM
            num_samples: Number of samples for Chronos
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
                - 'forecast': Ensemble point forecast (H,)
                - 'quantiles': Quantiles (if Chronos available)
                - 'individual_forecasts': Individual model outputs
                - 'uncertainty': Uncertainty estimate
                - 'ensemble_strategy': Strategy used
        """
        individual_forecasts = {}

        # Get forecasts from all models
        for name, model in self.models.items():
            try:
                if name == 'chronos':
                    individual_forecasts[name] = model.forecast(
                        context=context,
                        horizon=horizon,
                        num_samples=num_samples,
                        **kwargs
                    )
                else:
                    individual_forecasts[name] = model.forecast(
                        context=context,
                        horizon=horizon,
                        **kwargs
                    )
            except Exception as e:
                print(f"Warning: {name} forecast failed: {e}")
                continue

        if not individual_forecasts:
            raise RuntimeError("All forecasting models failed")

        # Ensemble point forecast
        if strategy == 'average' and len(individual_forecasts) > 1:
            # Average all model forecasts
            forecasts = [f['forecast'] for f in individual_forecasts.values()]
            point_forecast = np.mean(forecasts, axis=0)

        elif strategy == 'chronos_only' and 'chronos' in individual_forecasts:
            point_forecast = individual_forecasts['chronos']['forecast']

        elif strategy == 'timesfm_only' and 'timesfm' in individual_forecasts:
            point_forecast = individual_forecasts['timesfm']['forecast']

        elif len(individual_forecasts) == 1:
            # Only one model available
            point_forecast = list(individual_forecasts.values())[0]['forecast']

        else:
            raise ValueError(f"Unknown strategy '{strategy}' or missing models")

        # Get quantiles (from Chronos if available)
        quantiles = None
        samples = None
        if 'chronos' in individual_forecasts:
            quantiles = individual_forecasts['chronos']['quantiles']
            samples = individual_forecasts['chronos'].get('samples')

        # Estimate uncertainty
        uncertainty = self._estimate_uncertainty(individual_forecasts, quantiles)

        return {
            'forecast': point_forecast,
            'quantiles': quantiles,
            'samples': samples,
            'individual_forecasts': individual_forecasts,
            'uncertainty': uncertainty,
            'ensemble_strategy': strategy,
            'models_used': list(individual_forecasts.keys())
        }

    def _estimate_uncertainty(
        self,
        individual_forecasts: Dict,
        quantiles: Optional[Dict]
    ) -> np.ndarray:
        """Estimate forecast uncertainty

        Args:
            individual_forecasts: Individual model outputs
            quantiles: Quantiles from Chronos (if available)

        Returns:
            Uncertainty estimate (H,)
        """
        # Method 1: Use Chronos quantiles (best)
        if quantiles is not None:
            # Use interquartile range (P90 - P10)
            return quantiles['P90'] - quantiles['P10']

        # Method 2: Use disagreement between models
        if len(individual_forecasts) > 1:
            forecasts = np.array([f['forecast'] for f in individual_forecasts.values()])
            # Standard deviation across models
            return np.std(forecasts, axis=0)

        # Method 3: No uncertainty estimate available
        horizon = len(list(individual_forecasts.values())[0]['forecast'])
        return np.zeros(horizon)

    def get_model_agreement(self, forecast_result: Dict) -> float:
        """Measure agreement between models

        Args:
            forecast_result: Output from forecast()

        Returns:
            Agreement score in [0, 1] (1 = perfect agreement)
        """
        individual = forecast_result['individual_forecasts']

        if len(individual) < 2:
            return 1.0  # Perfect agreement if only one model

        # Compute pairwise correlations
        forecasts = [f['forecast'] for f in individual.values()]
        correlations = []

        for i in range(len(forecasts)):
            for j in range(i + 1, len(forecasts)):
                corr = np.corrcoef(forecasts[i], forecasts[j])[0, 1]
                correlations.append(corr)

        # Average correlation as agreement score
        return float(np.mean(correlations))
