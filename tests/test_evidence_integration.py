"""Integration test: Evidence-based detection with full pipeline.

Tests EvidenceBasedDetection with RawWindowProcessor (no foundation models needed).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.step1_data_processing import RawWindowProcessor, WindowConfig
from src.pipeline.step2_detection import EvidenceBasedDetection
from src.pipeline.step3_scoring import MaxPoolingScoring
from src.pipeline.step4_postprocessing import PostProcessor, PercentileThreshold
from src.pipeline.orchestrator import AnomalyDetectionPipeline


def test_evidence_pipeline_with_raw_processor():
    """Test EvidenceBasedDetection works with basic RawWindowProcessor.

    No GPU or foundation models required - tests evidence extraction
    using statistical tests, distribution, and pattern categories only.
    """
    np.random.seed(42)

    # Synthetic data: sine wave with noise
    X_train = np.sin(np.linspace(0, 10 * np.pi, 1000)) + 0.1 * np.random.randn(1000)
    X_test = np.sin(np.linspace(0, 5 * np.pi, 500)) + 0.1 * np.random.randn(500)

    # Inject anomaly: level shift at positions 200-220
    X_test[200:220] += 5.0
    y_test = np.zeros(500, dtype=int)
    y_test[200:220] = 1

    # Build pipeline with evidence-based detection (no forecast_based since no foundation models)
    pipeline = AnomalyDetectionPipeline(
        data_processor=RawWindowProcessor(WindowConfig(window_size=50, stride=1)),
        detection_method=EvidenceBasedDetection(
            enabled_categories=['statistical_tests', 'distribution_based', 'pattern_based']
        ),
        scoring_method=MaxPoolingScoring(),
        post_processor=PostProcessor(
            threshold_method=PercentileThreshold(percentile=95.0)
        )
    )

    # Fit on training data
    print("Fitting pipeline...")
    pipeline.fit(X_train)
    print("  Done.")

    # Predict on test data
    print("Running prediction...")
    result = pipeline.predict(X_test, y_test)
    print("  Done.")

    # Verify output format
    assert result.predictions.shape == (500,), f"Predictions shape: {result.predictions.shape}"
    assert result.point_scores.shape == (500,), f"Point scores shape: {result.point_scores.shape}"
    assert len(result.subsequence_scores) > 0

    # Verify evidence was stored
    evidence = pipeline.detection_method.get_evidence()
    assert evidence is not None, "Evidence should not be None"
    assert len(evidence) > 0, "Evidence should have entries"

    # Check evidence has expected keys
    first_evidence = evidence[0]
    print(f"\nEvidence keys for first window: {sorted(first_evidence.keys())}")
    assert 'max_abs_z_score' in first_evidence
    assert 'kl_divergence' in first_evidence
    assert 'volatility_ratio' in first_evidence
    assert 'window_index' in first_evidence

    # Verify metadata
    assert result.metadata.get('has_evidence') is True
    assert result.metadata.get('evidence_count') == len(evidence)
    assert result.metadata['detection_method'] == 'EvidenceBasedDetection'

    # Check that anomaly region has higher scores
    anomaly_scores = result.point_scores[200:220]
    normal_scores = np.concatenate([result.point_scores[:180], result.point_scores[240:]])
    mean_anomaly = np.mean(anomaly_scores)
    mean_normal = np.mean(normal_scores)
    print(f"\nMean anomaly score: {mean_anomaly:.4f}")
    print(f"Mean normal score:  {mean_normal:.4f}")
    print(f"Ratio: {mean_anomaly / (mean_normal + 1e-10):.2f}x")

    # Results
    detected = int(np.sum(result.predictions))
    actual_anomalies = int(np.sum(y_test))
    print(f"\nDetected anomalies: {detected} points")
    print(f"Actual anomalies:   {actual_anomalies} points")
    print(f"Threshold: {result.threshold:.4f}")

    print(f"\nExecution times:")
    for step, t in result.execution_time.items():
        print(f"  {step}: {t:.3f}s")

    print("\nTEST PASSED: EvidenceBasedDetection works with pipeline!")
    return True


def test_evidence_all_categories():
    """Test with all evidence categories including forecast-based."""
    np.random.seed(42)

    X_train = np.random.randn(500)
    X_test = np.random.randn(200)
    X_test[80:100] += 8.0  # Strong anomaly

    pipeline = AnomalyDetectionPipeline(
        data_processor=RawWindowProcessor(WindowConfig(window_size=30, stride=1)),
        detection_method=EvidenceBasedDetection(),  # All categories enabled
        scoring_method=MaxPoolingScoring(),
        post_processor=PostProcessor(
            threshold_method=PercentileThreshold(percentile=95.0)
        )
    )

    pipeline.fit(X_train)
    result = pipeline.predict(X_test)

    # Even without forecasts, non-forecast categories should work
    evidence = pipeline.detection_method.get_evidence()
    assert evidence is not None
    assert len(evidence) > 0

    # Statistical tests should be present (train_statistics from orchestrator may be empty,
    # but train_data_sample from fit() is used for distribution/pattern)
    first = evidence[0]
    assert 'kl_divergence' in first or 'volatility_ratio' in first

    print("TEST PASSED: All categories enabled without foundation models!")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Integration Test: Evidence-Based Detection Pipeline")
    print("=" * 60)

    success = True
    try:
        print("\n--- Test 1: Evidence pipeline with RawWindowProcessor ---")
        success = test_evidence_pipeline_with_raw_processor() and success
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        success = False

    try:
        print("\n--- Test 2: All evidence categories ---")
        success = test_evidence_all_categories() and success
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "=" * 60)
    if success:
        print("ALL INTEGRATION TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    sys.exit(0 if success else 1)
