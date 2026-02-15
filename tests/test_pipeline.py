"""Tests for anomaly detection pipeline"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import (
    WindowConfig,
    RawWindowProcessor,
    DistanceBasedDetection,
    MaxPoolingScoring,
    PercentileThreshold,
    PostProcessor,
    AnomalyDetectionPipeline
)
from src.data.loader import create_synthetic_dataset
from src.evaluation.evaluator import Evaluator


def test_synthetic_data_generation():
    """Test synthetic dataset generation"""
    dataset = create_synthetic_dataset(n_samples=1000, n_dims=3, anomaly_ratio=0.05)

    # Check shapes
    assert dataset.X_train.shape[1] == 3
    assert dataset.X_val.shape[1] == 3
    assert dataset.X_test.shape[1] == 3

    # Check labels
    assert dataset.y_train.sum() == 0  # Training should be all normal
    assert dataset.y_test.sum() > 0  # Test should have anomalies


def test_data_processor():
    """Test data processor"""
    X = np.random.randn(200, 3)

    processor = RawWindowProcessor(WindowConfig(window_size=50, stride=1))
    X_processed = processor.process(X, fit=True)

    # Should create windows
    assert X_processed.ndim == 3
    assert X_processed.shape[1] == 50  # Window size
    assert X_processed.shape[2] == 3  # Dimensions


def test_detection_method():
    """Test detection method"""
    X_train = np.random.randn(100, 50, 3)
    X_test = np.random.randn(20, 50, 3)

    detector = DistanceBasedDetection(k=5)
    detector.fit(X_train)
    scores = detector.detect(X_test)

    # Should return one score per window
    assert scores.shape == (20,)
    assert np.all(scores >= 0)  # Distances should be non-negative


def test_scoring_method():
    """Test scoring method"""
    subsequence_scores = np.array([1.0, 2.0, 3.0])
    window_size = 50
    stride = 25
    original_length = 100

    scorer = MaxPoolingScoring()
    point_scores = scorer.score(subsequence_scores, window_size, stride, original_length)

    # Should return point-wise scores
    assert point_scores.shape == (original_length,)


def test_post_processor():
    """Test post-processor"""
    scores = np.random.rand(100)
    labels = np.random.randint(0, 2, 100)

    post_processor = PostProcessor(
        threshold_method=PercentileThreshold(percentile=95),
        min_anomaly_length=3,
        merge_gap=5
    )

    predictions, threshold = post_processor.process(scores, labels)

    # Should return binary predictions
    assert predictions.shape == scores.shape
    assert np.all((predictions == 0) | (predictions == 1))
    assert threshold > 0


def test_end_to_end_pipeline():
    """Test complete pipeline end-to-end"""
    # Create small dataset
    dataset = create_synthetic_dataset(n_samples=500, n_dims=3, anomaly_ratio=0.05)

    # Build pipeline
    pipeline = AnomalyDetectionPipeline(
        data_processor=RawWindowProcessor(WindowConfig(window_size=50, stride=1)),
        detection_method=DistanceBasedDetection(k=5),
        scoring_method=MaxPoolingScoring(),
        post_processor=PostProcessor(
            threshold_method=PercentileThreshold(percentile=95),
            min_anomaly_length=1,
            merge_gap=0
        )
    )

    # Train
    pipeline.fit(dataset.X_train)

    # Predict
    result = pipeline.predict(dataset.X_test)

    # Check outputs
    assert result.predictions.shape == (len(dataset.X_test),)
    assert result.point_scores.shape == (len(dataset.X_test),)
    assert len(result.subsequence_scores) > 0
    assert result.threshold > 0

    # Check timing recorded
    assert 'total' in result.execution_time
    assert result.execution_time['total'] > 0


def test_evaluation():
    """Test evaluation metrics"""
    y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    scores = np.random.rand(10)

    evaluator = Evaluator()
    result = evaluator.evaluate(y_true, y_pred, scores)

    # Check metrics exist
    assert 0 <= result.f1 <= 1
    assert 0 <= result.precision <= 1
    assert 0 <= result.recall <= 1
    assert 0 <= result.pa_f1 <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
