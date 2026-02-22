"""Loader for AnomLLM synthetic dataset"""

import pickle
import numpy as np
from typing import Dict, List, Tuple
from .loader import Dataset


def load_anomllm_category(category: str, base_path: str = "src/data/synthetic") -> Dataset:
    """Load AnomLLM dataset for a specific category

    Categories: 'point', 'trend', 'range', 'freq', 'noisy-point', 'noisy-trend',
                'noisy-freq', 'flat-trend'

    Args:
        category: Category name (e.g., 'point', 'trend')
        base_path: Base path to synthetic data

    Returns:
        Dataset with train/val/test splits
    """
    import os

    # Load train data
    train_path = os.path.join(base_path, category, "train", "data.pkl")
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    # Load eval data
    eval_path = os.path.join(base_path, category, "eval", "data.pkl")
    with open(eval_path, 'rb') as f:
        eval_data = pickle.load(f)

    # Process train data (combine all series)
    X_train_list = []
    y_train_list = []

    for series, anom_segments in zip(train_data['series'], train_data['anom']):
        # series: (T, D) numpy array
        # anom_segments: [[(start, end), ...]]
        T = len(series)
        labels = np.zeros(T, dtype=int)

        # Mark anomaly segments
        for segment_list in anom_segments:
            for start, end in segment_list:
                labels[start:end] = 1

        X_train_list.append(series)
        y_train_list.append(labels)

    # Concatenate all train series
    X_train = np.vstack(X_train_list)  # (N*T, D)
    y_train = np.concatenate(y_train_list)  # (N*T,)

    # Process eval data (combine all series)
    X_eval_list = []
    y_eval_list = []

    for series, anom_segments in zip(eval_data['series'], eval_data['anom']):
        T = len(series)
        labels = np.zeros(T, dtype=int)

        # Mark anomaly segments
        for segment_list in anom_segments:
            for start, end in segment_list:
                labels[start:end] = 1

        X_eval_list.append(series)
        y_eval_list.append(labels)

    # Concatenate all eval series
    X_eval = np.vstack(X_eval_list)  # (N*T, D)
    y_eval = np.concatenate(y_eval_list)  # (N*T,)

    # Split eval into validation and test (50/50)
    split_idx = len(X_eval) // 2
    X_val = X_eval[:split_idx]
    y_val = y_eval[:split_idx]
    X_test = X_eval[split_idx:]
    y_test = y_eval[split_idx:]

    # For unsupervised training, set train labels to all normal
    y_train_clean = np.zeros_like(y_train)

    return Dataset(
        X_train=X_train,
        y_train=y_train_clean,  # All normal for unsupervised
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        name=f"anomllm_{category}",
        metadata={
            "source": "AnomLLM",
            "category": category,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "n_dims": X_train.shape[1],
            "train_anomalies": int(y_train.sum()),
            "val_anomalies": int(y_val.sum()),
            "test_anomalies": int(y_test.sum()),
            "train_series_count": len(train_data['series']),
            "eval_series_count": len(eval_data['series'])
        }
    )


def load_anomllm_series(category: str, split: str = "eval",
                        base_path: str = "src/data/synthetic") -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load AnomLLM dataset as individual series (not concatenated).

    This matches AnomLLM's per-series evaluation approach.

    Args:
        category: Category name (e.g., 'point', 'trend')
        split: 'train' or 'eval'
        base_path: Base path to synthetic data

    Returns:
        List of (series, labels) tuples where:
            series: np.ndarray (T, D) — time series
            labels: np.ndarray (T,) — binary labels
    """
    import os

    data_path = os.path.join(base_path, category, split, "data.pkl")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    result = []
    for series, anom_segments in zip(data['series'], data['anom']):
        T = len(series)
        labels = np.zeros(T, dtype=int)
        for segment_list in anom_segments:
            for start, end in segment_list:
                labels[start:end] = 1
        result.append((series, labels))

    return result


def get_all_categories() -> List[str]:
    """Get list of all available AnomLLM categories"""
    return [
        'point',
        'trend',
        'range',
        'freq',
        'noisy-point',
        'noisy-trend',
        'noisy-freq',
        'flat-trend'
    ]


def load_all_anomllm_categories(base_path: str = "src/data/synthetic") -> Dict[str, Dataset]:
    """Load all AnomLLM categories

    Returns:
        Dictionary mapping category name to Dataset
    """
    datasets = {}
    categories = get_all_categories()

    for category in categories:
        try:
            datasets[category] = load_anomllm_category(category, base_path)
            print(f"[OK] Loaded {category}: {datasets[category].metadata['test_size']} test samples, "
                  f"{datasets[category].metadata['test_anomalies']} anomalies")
        except Exception as e:
            print(f"[FAIL] Failed to load {category}: {e}")

    return datasets
