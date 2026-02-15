"""Dataset loaders for time series anomaly detection"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class Dataset:
    """Dataset container"""
    X_train: np.ndarray  # (T_train, D)
    y_train: np.ndarray  # (T_train,)
    X_val: np.ndarray  # (T_val, D)
    y_val: np.ndarray  # (T_val,)
    X_test: np.ndarray  # (T_test, D)
    y_test: np.ndarray  # (T_test,)
    name: str
    metadata: Dict[str, Any]


def create_synthetic_dataset(n_samples: int = 1000,
                            n_dims: int = 5,
                            anomaly_ratio: float = 0.05,
                            random_seed: int = 42) -> Dataset:
    """Create synthetic dataset for testing

    Normal data: Sine waves + Gaussian noise
    Anomalies: Random spikes, level shifts, or trend changes

    Args:
        n_samples: Total number of timesteps
        n_dims: Number of dimensions (channels)
        anomaly_ratio: Fraction of anomalous points (0-1)
        random_seed: Random seed for reproducibility

    Returns:
        Dataset with train/val/test splits
    """
    np.random.seed(random_seed)

    # Generate time axis
    t = np.linspace(0, 10 * np.pi, n_samples)

    # Generate normal data (sine waves with different frequencies + noise)
    X = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        freq = 1.0 + d * 0.5  # Different frequency for each dimension
        phase = np.random.uniform(0, 2 * np.pi)
        X[:, d] = np.sin(freq * t + phase) + np.random.normal(0, 0.1, n_samples)

    # Generate anomalies
    y = np.zeros(n_samples, dtype=int)
    n_anomalies = int(n_samples * anomaly_ratio)

    # Create anomaly segments (3-10 points each)
    anomaly_starts = np.random.choice(
        range(n_samples - 10),
        size=n_anomalies // 5,
        replace=False
    )

    for start in anomaly_starts:
        length = np.random.randint(3, 10)
        end = min(start + length, n_samples)

        # Mark as anomaly
        y[start:end] = 1

        # Inject anomaly (choose random type)
        anomaly_type = np.random.choice(['spike', 'shift', 'trend'])

        if anomaly_type == 'spike':
            # Random spike
            X[start:end] += np.random.normal(0, 3, (end - start, n_dims))

        elif anomaly_type == 'shift':
            # Level shift
            shift = np.random.normal(2, 0.5, n_dims)
            X[start:end] += shift

        elif anomaly_type == 'trend':
            # Trend change
            trend = np.linspace(0, 3, end - start)
            X[start:end] += trend[:, np.newaxis]

    # Split into train/val/test (60/20/20)
    train_end = int(0.6 * n_samples)
    val_end = int(0.8 * n_samples)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    # Training data should be mostly normal (set anomaly labels to 0)
    y_train = np.zeros_like(y_train)

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        name="synthetic",
        metadata={
            "n_samples": n_samples,
            "n_dims": n_dims,
            "anomaly_ratio": anomaly_ratio,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "test_anomalies": int(y_test.sum())
        }
    )
