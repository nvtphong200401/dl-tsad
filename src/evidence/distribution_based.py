"""Distribution-Based Evidence Extraction

Compares distributions between training and test windows.
Metrics: KL divergence, Wasserstein distance.
"""

import numpy as np
from typing import Dict
from scipy.stats import entropy, wasserstein_distance


class DistributionBasedEvidence:
    """Extract distribution comparison metrics.

    Metrics:
        1. KL divergence (histogram-based)
        2. Wasserstein distance (Earth Mover's Distance)
    """

    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins

    def extract(
        self,
        train_data: np.ndarray,
        test_window: np.ndarray
    ) -> Dict:
        """Extract distribution-based evidence.

        Args:
            train_data: Representative training data (M,)
            test_window: Test window values (W,)

        Returns:
            Dict with distribution comparison metrics.
        """
        evidence = {}
        train = train_data.flatten().astype(float)
        test = test_window.flatten().astype(float)

        if len(train) < 2 or len(test) < 2:
            return evidence

        # 1. KL divergence
        self._compute_kl_divergence(train, test, evidence)

        # 2. Wasserstein distance
        self._compute_wasserstein(train, test, evidence)

        return evidence

    def _compute_kl_divergence(
        self,
        train: np.ndarray,
        test: np.ndarray,
        evidence: Dict
    ) -> None:
        """Compute KL divergence between train and test distributions."""
        # Use bin edges from the combined range
        combined = np.concatenate([train, test])
        bin_edges = np.linspace(
            np.min(combined) - 1e-10,
            np.max(combined) + 1e-10,
            self.n_bins + 1
        )

        hist_train, _ = np.histogram(train, bins=bin_edges, density=True)
        hist_test, _ = np.histogram(test, bins=bin_edges, density=True)

        # Add epsilon to avoid log(0)
        eps = 1e-10
        hist_train = hist_train + eps
        hist_test = hist_test + eps

        # Normalize to proper distributions
        hist_train = hist_train / hist_train.sum()
        hist_test = hist_test / hist_test.sum()

        kl_div = float(entropy(hist_train, hist_test))
        evidence['kl_divergence'] = kl_div

    def _compute_wasserstein(
        self,
        train: np.ndarray,
        test: np.ndarray,
        evidence: Dict
    ) -> None:
        """Compute Wasserstein (Earth Mover's) distance."""
        w_dist = float(wasserstein_distance(train, test))
        train_std = float(np.std(train))
        std_safe = train_std if train_std > 1e-10 else 1e-10

        evidence['wasserstein_distance'] = w_dist
        evidence['normalized_wasserstein'] = w_dist / std_safe
