"""Step 1: Data Processing - Window transformation and preprocessing"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class WindowConfig:
    """Configuration for window transformation"""
    window_size: int = 100
    stride: int = 1
    padding: str = "same"  # "same", "valid", "causal"


class DataProcessor(ABC):
    """Base class for Step 1: Data Processing

    All data processors must:
    1. Transform time series into sliding windows
    2. Apply optional preprocessing
    3. Support fit/transform paradigm
    """

    def __init__(self, window_config: WindowConfig):
        self.window_config = window_config
        self.is_fitted = False

    def process(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Full data processing pipeline

        Args:
            X: Time series data (T, D) - T timesteps, D dimensions
            fit: Whether to fit preprocessing parameters

        Returns:
            Processed windows (N, W, D') - N windows, W window size, D' processed dims
        """
        # Step 1a: Window transformation (mandatory)
        windows = self._create_windows(X)

        # Step 1b: Pre-processing (optional, varies by method)
        if fit:
            processed = self.fit_transform(windows)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Must fit before transform")
            processed = self.transform(windows)

        return processed

    def _create_windows(self, X: np.ndarray) -> np.ndarray:
        """Create sliding windows from time series

        Args:
            X: Time series (T, D)

        Returns:
            Windows (N, W, D)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        T, D = X.shape
        W = self.window_config.window_size
        S = self.window_config.stride

        windows = []
        for i in range(0, T - W + 1, S):
            windows.append(X[i:i + W])

        return np.array(windows)  # Shape: (N, W, D)

    @abstractmethod
    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """Fit preprocessing and transform windows

        Args:
            windows: Windowed data (N, W, D)

        Returns:
            Processed windows (N, W, D')
        """
        pass

    @abstractmethod
    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Transform windows using fitted preprocessing

        Args:
            windows: Windowed data (N, W, D)

        Returns:
            Processed windows (N, W, D')
        """
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return dimension of processed output"""
        pass


class RawWindowProcessor(DataProcessor):
    """Simplest processor: Just window transformation + z-score normalization

    This is the baseline processor that:
    1. Creates sliding windows
    2. Normalizes each window using StandardScaler
    3. Outputs normalized windows
    """

    def __init__(self, window_config: WindowConfig):
        super().__init__(window_config)
        self.scaler = StandardScaler()
        self.input_dim = None

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """Fit scaler and normalize windows"""
        N, W, D = windows.shape
        self.input_dim = D

        # Reshape to (N*W, D) for normalization
        windows_flat = windows.reshape(-1, D)

        # Fit and transform
        normalized = self.scaler.fit_transform(windows_flat)

        # Reshape back to (N, W, D)
        return normalized.reshape(N, W, D)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Normalize windows using fitted scaler"""
        N, W, D = windows.shape

        # Reshape to (N*W, D)
        windows_flat = windows.reshape(-1, D)

        # Transform
        normalized = self.scaler.transform(windows_flat)

        # Reshape back
        return normalized.reshape(N, W, D)

    def get_output_dim(self) -> int:
        """Output dimension is same as input"""
        return self.input_dim if self.input_dim else 0
