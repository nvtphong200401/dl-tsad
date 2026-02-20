"""Base class for foundation models"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class FoundationModel(ABC):
    """Base class for foundation forecasting models"""

    @abstractmethod
    def load_model(self):
        """Load pre-trained model"""
        pass

    @abstractmethod
    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate forecast

        Args:
            context: Historical data to use as context (T,) or (T, D)
            horizon: Number of steps to forecast
            **kwargs: Model-specific parameters

        Returns:
            Dictionary with:
                - 'forecast': Point forecast (H,) or (H, D)
                - 'model': Model identifier string
                - Additional model-specific outputs
        """
        pass

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return hasattr(self, 'model') and self.model is not None
