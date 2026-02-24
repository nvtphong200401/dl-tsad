"""STL-based Data Processor for Step 1

Fast alternative to foundation model processors (Chronos, TimesFM).
Uses STL decomposition to generate forecasts, quantiles, and samples
for downstream evidence extraction — all on CPU in milliseconds.
"""

import numpy as np
from typing import Dict, Optional, List
from .step1_data_processing import DataProcessor, WindowConfig


class STLProcessor(DataProcessor):
    """STL decomposition-based data processor.

    Replaces FoundationModelProcessor with a fast, CPU-only approach:
    1. Create sliding windows (inherited)
    2. Normalize windows
    3. Decompose series via STL (trend + seasonal + residual)
    4. Generate per-window forecasts by extrapolating trend + seasonal
    5. Estimate uncertainty via residual bootstrap

    Provides the same get_forecasts() / get_train_statistics() interface
    as FoundationModelProcessor for seamless orchestrator integration.
    """

    def __init__(
        self,
        window_config: WindowConfig,
        forecast_horizon: Optional[int] = None,
        period: Optional[int] = None,
        seasonal: int = 7,
        trend: Optional[int] = None,
        robust: bool = True,
        num_synthetic_samples: int = 50,
    ):
        """Initialize STL processor.

        Args:
            window_config: Window configuration (size, stride)
            forecast_horizon: Steps to forecast per window (default: window_size)
            period: Seasonal period (auto-detect via ACF if None)
            seasonal: STL seasonal smoother span (must be odd, >= 7)
            trend: STL trend smoother span (None = statsmodels default)
            robust: Use robust STL fitting (downweights outliers)
            num_synthetic_samples: Number of bootstrap samples for quantiles
        """
        super().__init__(window_config)

        self.forecast_horizon = forecast_horizon or window_config.window_size
        self.period = period
        self.seasonal = seasonal
        self.trend_smoother = trend
        self.robust = robust
        self.num_synthetic_samples = num_synthetic_samples

        # Set during fit_transform()
        self.train_mean: Optional[np.ndarray] = None
        self.train_std: Optional[np.ndarray] = None
        self.train_statistics: Optional[Dict] = None
        self.detected_period: Optional[int] = None
        self.seasonal_pattern: Optional[np.ndarray] = None
        self.trend_slope: Optional[float] = None
        self.trend_intercept: Optional[float] = None
        self.residual_std: Optional[float] = None
        self.residual_distribution: Optional[np.ndarray] = None

        # Set during transform()
        self.forecast_results: Optional[List[Dict]] = None

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """Fit STL on training windows, compute statistics.

        Args:
            windows: Windowed training data (N, W, D)

        Returns:
            Normalized windows (N, W, D)
        """
        from statsmodels.tsa.seasonal import STL

        N, W, D = windows.shape

        # Compute normalization and training statistics
        windows_flat = windows.reshape(-1, D)
        self.train_mean = np.mean(windows_flat, axis=0)
        self.train_std = np.std(windows_flat, axis=0) + 1e-10

        self.train_statistics = {
            'mean': self.train_mean,
            'std': self.train_std,
            'quantiles': {
                f'P{int(q * 100)}': np.quantile(windows_flat, q, axis=0)
                for q in [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
            },
            'min': np.min(windows_flat, axis=0),
            'max': np.max(windows_flat, axis=0),
        }

        # Reconstruct full series for STL (dimension 0)
        full_series = self._reconstruct_series(windows[:, :, 0])

        # Detect or validate seasonal period
        if self.period is not None:
            self.detected_period = self.period
        else:
            self.detected_period = self._detect_period(full_series)

        # Guard: period must be valid for STL
        min_period = 2
        max_period = len(full_series) // 2 - 1
        if self.detected_period < min_period or self.detected_period > max_period:
            self.detected_period = max(min_period, min(W // 4, max_period))

        # Run STL decomposition
        stl_result = STL(
            full_series,
            period=self.detected_period,
            seasonal=self._ensure_odd(max(self.seasonal, 7)),
            trend=self.trend_smoother,
            robust=self.robust,
        ).fit()

        # Store seasonal pattern (one full cycle)
        self.seasonal_pattern = stl_result.seasonal[:self.detected_period]

        # Fit linear model to trend for extrapolation
        trend_component = stl_result.trend
        t_indices = np.arange(len(trend_component))
        valid = ~np.isnan(trend_component)
        if np.sum(valid) >= 2:
            self.trend_slope, self.trend_intercept = np.polyfit(
                t_indices[valid], trend_component[valid], 1
            )
        else:
            self.trend_slope = 0.0
            self.trend_intercept = float(np.nanmean(full_series))

        # Compute actual in-window forecast errors on training data.
        # This captures the real error between the STL model and actual values,
        # which is larger than just the STL residual (since the model will be
        # applied to unseen test data with additional mismatch).
        S = self.window_config.stride
        T_full = len(full_series)
        t_indices_full = np.arange(T_full)
        expected_train = (self.trend_slope * t_indices_full + self.trend_intercept
                          + self._extrapolate_seasonal(0, T_full))
        forecast_errors = full_series - expected_train
        valid_errors = forecast_errors[~np.isnan(forecast_errors)]
        if len(valid_errors) > 0:
            self.residual_std = float(np.std(valid_errors))
            self.residual_distribution = valid_errors.copy()
        else:
            self.residual_std = float(self.train_std[0])
            self.residual_distribution = np.zeros(10)

        # Normalize and return
        normalized = (windows_flat - self.train_mean) / self.train_std
        normalized = normalized.reshape(N, W, D)

        print(f"  STL processor fitted on {N} windows (period={self.detected_period})")

        return normalized

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Transform test windows and generate in-window STL forecasts.

        Instead of forward-looking forecasts, generates expected values
        for each window based on STL decomposition (trend + seasonal).
        This works for ALL window positions — no blind spots at edges.

        Args:
            windows: Windowed test data (N, W, D)

        Returns:
            Normalized windows (N, W, D)

        Side Effect:
            Stores forecasts in self.forecast_results for Step 2.
        """
        from statsmodels.tsa.seasonal import STL

        if self.train_mean is None:
            raise ValueError("Must call fit_transform() before transform()")

        N, W, D = windows.shape
        S = self.window_config.stride

        # Normalize
        windows_flat = windows.reshape(-1, D)
        normalized = (windows_flat - self.train_mean) / self.train_std
        normalized = normalized.reshape(N, W, D)

        # Reconstruct full test series for STL
        full_series = self._reconstruct_series(windows[:, :, 0])

        # Build expected series from TRAINING model (not test STL).
        # Using training trend + seasonal ensures anomalies in the test
        # series produce large deviations from the expected baseline.
        T_full = len(full_series)
        t_indices = np.arange(T_full)
        expected_trend = self.trend_slope * t_indices + self.trend_intercept
        expected_seasonal = self._extrapolate_seasonal(0, T_full)
        expected_series = expected_trend + expected_seasonal

        # Generate per-window in-window forecasts
        self.forecast_results = []

        for i in range(N):
            window_start = i * S
            window_end = window_start + W

            # In-window forecast: what this window SHOULD look like
            expected_window = expected_series[window_start:window_end]

            # Handle edge case where expected is shorter than W
            if len(expected_window) < W:
                padded = np.full(W, expected_window[-1] if len(expected_window) > 0 else 0.0)
                padded[:len(expected_window)] = expected_window
                expected_window = padded

            # Normalize to match normalized window space
            forecast_norm = (expected_window - self.train_mean[0]) / self.train_std[0]

            # Generate bootstrap samples for uncertainty
            samples = self._generate_samples(forecast_norm, W)

            # Compute quantiles from samples
            quantiles = {
                'P01': np.quantile(samples, 0.01, axis=0),
                'P10': np.quantile(samples, 0.10, axis=0),
                'P25': np.quantile(samples, 0.25, axis=0),
                'P50': np.quantile(samples, 0.50, axis=0),
                'P75': np.quantile(samples, 0.75, axis=0),
                'P90': np.quantile(samples, 0.90, axis=0),
                'P99': np.quantile(samples, 0.99, axis=0),
            }

            uncertainty = quantiles['P90'] - quantiles['P10']

            self.forecast_results.append({
                'forecast': forecast_norm,
                'quantiles': quantiles,
                'samples': samples,
                'uncertainty': uncertainty,
            })

        print(f"  STL generated {len(self.forecast_results)} in-window forecasts")

        return normalized

    def get_output_dim(self) -> int:
        return 1 if self.train_mean is None else len(self.train_mean)

    def get_forecasts(self) -> Optional[List[Dict]]:
        """Get stored forecast results from transform()."""
        return self.forecast_results

    def get_train_statistics(self) -> Optional[Dict]:
        """Get training statistics from fit_transform()."""
        return self.train_statistics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reconstruct_series(self, windows_1d: np.ndarray) -> np.ndarray:
        """Reconstruct full 1D series from overlapping windows.

        Args:
            windows_1d: (N, W) single-dimension windows

        Returns:
            Reconstructed series (T,) via overlap averaging
        """
        N, W = windows_1d.shape
        S = self.window_config.stride
        T = (N - 1) * S + W

        series = np.zeros(T)
        counts = np.zeros(T)

        for i in range(N):
            start = i * S
            series[start:start + W] += windows_1d[i]
            counts[start:start + W] += 1

        return series / np.maximum(counts, 1)

    def _detect_period(self, series: np.ndarray) -> int:
        """Auto-detect seasonal period via ACF peak detection.

        Args:
            series: 1D time series

        Returns:
            Detected period (>= 2)
        """
        from scipy.signal import find_peaks

        n = len(series)
        max_lag = min(n // 2, 500)

        mean = np.mean(series)
        var = np.var(series)
        if var < 1e-10:
            return max(2, n // 10)

        # Compute autocorrelation
        centered = series - mean
        acf = np.correlate(centered, centered, mode='full')
        acf = acf[n - 1:]  # positive lags only
        acf = acf / (var * n)
        acf = acf[:max_lag]

        # Find first significant peak
        peaks, _ = find_peaks(acf[1:], height=0.1)

        if len(peaks) > 0:
            return int(peaks[0] + 1)
        else:
            return max(2, n // 10)

    def _extrapolate_seasonal(self, start_pos: int, H: int) -> np.ndarray:
        """Extrapolate seasonal component H steps from start_pos.

        Phase-aligned tiling of the stored seasonal pattern.
        """
        period = self.detected_period
        phase = start_pos % period
        n_tiles = (H // period) + 2
        tiled = np.tile(self.seasonal_pattern, n_tiles)
        return tiled[phase:phase + H]

    def _generate_samples(
        self, point_forecast: np.ndarray, H: int
    ) -> np.ndarray:
        """Generate bootstrap forecast samples for uncertainty.

        Adds random residuals (from training) to point forecast.

        Returns:
            (num_synthetic_samples, H) array
        """
        norm_residuals = self.residual_distribution / (self.train_std[0] + 1e-10)

        samples = np.zeros((self.num_synthetic_samples, H))
        for s in range(self.num_synthetic_samples):
            noise = np.random.choice(norm_residuals, size=H, replace=True)
            samples[s] = point_forecast + noise

        return samples

    @staticmethod
    def _ensure_odd(n: int) -> int:
        """Ensure a number is odd (required by STL seasonal smoother)."""
        return n if n % 2 == 1 else n + 1
