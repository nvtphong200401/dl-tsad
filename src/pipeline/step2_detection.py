"""Step 2: Detection Method - Compute anomaly scores for windows"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors


class DetectionMethod(ABC):
    """Base class for Step 2: Detection

    Detection methods take processed windows and compute anomaly scores
    for each window (sub-sequence level).
    """

    @abstractmethod
    def fit(self, X_processed: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit detection method on processed training data

        Args:
            X_processed: Processed windows (N, W, D')
            y: Optional labels (N,)
        """
        pass

    @abstractmethod
    def detect(self, X_processed: np.ndarray) -> np.ndarray:
        """Return sub-sequence level anomaly scores

        Args:
            X_processed: Processed windows (N, W, D')

        Returns:
            Anomaly scores (N,) - one score per window
        """
        pass


class DistanceBasedDetection(DetectionMethod):
    """Distance-based detection using K-Nearest Neighbors

    Stores normal training data and computes anomaly score as
    the average distance to k-nearest neighbors.
    """

    def __init__(self, k: int = 5, method: str = "knn"):
        """Initialize distance-based detection

        Args:
            k: Number of nearest neighbors
            method: "knn" (K-Nearest Neighbors)
        """
        self.k = k
        self.method = method
        self.train_data = None
        self.nbrs = None

    def fit(self, X_processed: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Store training data for distance comparison

        Args:
            X_processed: Processed windows (N, W, D')
            y: Optional labels (ignored for unsupervised)
        """
        # Flatten windows to vectors for distance computation
        N, W, D = X_processed.shape
        self.train_data = X_processed.reshape(N, -1)  # (N, W*D)

        # Fit k-NN model
        self.nbrs = NearestNeighbors(n_neighbors=self.k)
        self.nbrs.fit(self.train_data)

    def detect(self, X_processed: np.ndarray) -> np.ndarray:
        """Compute distance to k-nearest neighbors as anomaly score

        Args:
            X_processed: Processed windows (N, W, D')

        Returns:
            Anomaly scores (N,) - average distance to k-NN
        """
        if self.nbrs is None:
            raise ValueError("Must call fit() before detect()")

        # Flatten windows
        N, W, D = X_processed.shape
        X_flat = X_processed.reshape(N, -1)

        # Compute distances to k-nearest neighbors
        distances, _ = self.nbrs.kneighbors(X_flat)

        # Anomaly score = average distance to k-NN
        scores = np.mean(distances, axis=1)

        return scores
