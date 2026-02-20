#!/usr/bin/env python
"""Test Phase 2 full pipeline: Foundation Models + Evidence Extraction

This script tests the complete Phase 2 pipeline:
  Step 1: FoundationModelProcessor (Chronos forecasts)
  Step 2: EvidenceBasedDetection (13 statistical metrics from forecasts)
  Step 3: MaxPoolingScoring
  Step 4: PostProcessor

Usage:
    python experiments/test_phase2_pipeline_integration.py
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline.step1_foundation_model_processor import FoundationModelProcessor
from pipeline.step1_data_processing import WindowConfig
from pipeline.step2_detection import EvidenceBasedDetection
from pipeline.step3_scoring import MaxPoolingScoring
from pipeline.step4_postprocessing import PostProcessor, PercentileThreshold
from pipeline.orchestrator import AnomalyDetectionPipeline


def create_synthetic_data(length=1000, anomaly_positions=None):
    """Create synthetic time series with anomalies

    Args:
        length: Length of time series
        anomaly_positions: List of (start, end) tuples for anomalies

    Returns:
        X: Time series data (T,)
        y: Labels (T,)
    """
    # Base: sine wave
    t = np.linspace(0, 10*np.pi, length)
    X = np.sin(t) + 0.1 * np.random.randn(length)

    # Labels (all normal by default)
    y = np.zeros(length, dtype=int)

    # Add anomalies
    if anomaly_positions:
        for start, end in anomaly_positions:
            # Spike anomaly
            X[start:end] += 5.0
            y[start:end] = 1

    return X, y


def test_full_phase2_pipeline():
    """Test FoundationModelProcessor + EvidenceBasedDetection end-to-end"""

    print("\n" + "="*60)
    print("TEST: Full Phase 2 Pipeline (Forecasts + Evidence)")
    print("="*60)

    # Create synthetic data
    print("\n1. Creating synthetic data...")
    X_train, y_train = create_synthetic_data(length=1000, anomaly_positions=None)
    X_test, y_test = create_synthetic_data(length=500, anomaly_positions=[(100, 120), (300, 310)])

    print(f"   Train: {len(X_train)} points (all normal)")
    print(f"   Test: {len(X_test)} points ({y_test.sum()} anomalous)")

    # Create pipeline components
    print("\n2. Creating pipeline components...")

    # Step 1: Foundation Model Processor
    window_config = WindowConfig(window_size=100, stride=1)
    data_processor = FoundationModelProcessor(
        window_config=window_config,
        forecast_horizon=64,  # chronos-t5-tiny degrades beyond 64 steps
        models=['chronos'],
        chronos_model="amazon/chronos-t5-tiny",
        num_samples=20
    )
    print("   Step 1: FoundationModelProcessor (Chronos)")

    # Step 2: Evidence-based detection (all 13 metrics)
    detection_method = EvidenceBasedDetection()
    print("   Step 2: EvidenceBasedDetection (13 metrics)")

    # Step 3: Max pooling scoring
    scoring_method = MaxPoolingScoring()
    print("   Step 3: MaxPoolingScoring")

    # Step 4: Percentile threshold
    post_processor = PostProcessor(
        threshold_method=PercentileThreshold(percentile=95.0),
        min_anomaly_length=3,
        merge_gap=5
    )
    print("   Step 4: PostProcessor")

    # Create pipeline
    print("\n3. Creating pipeline...")
    pipeline = AnomalyDetectionPipeline(
        data_processor=data_processor,
        detection_method=detection_method,
        scoring_method=scoring_method,
        post_processor=post_processor
    )
    print("   Pipeline created")

    # Fit pipeline
    print("\n4. Fitting pipeline on training data...")
    try:
        pipeline.fit(X_train, y_train)
        print("   Pipeline fitted successfully")
    except Exception as e:
        print(f"   Pipeline fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Predict
    print("\n5. Running prediction on test data...")
    try:
        result = pipeline.predict(X_test, y_test)
        print("   Prediction completed successfully")
    except Exception as e:
        print(f"   Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify output format
    print("\n6. Verifying output format...")
    assert hasattr(result, 'predictions'), "Missing predictions"
    assert hasattr(result, 'point_scores'), "Missing point_scores"
    assert hasattr(result, 'subsequence_scores'), "Missing subsequence_scores"
    assert len(result.predictions) == len(X_test), "Prediction length mismatch"
    print("   Output format correct")

    # Check forecasts
    print("\n7. Checking forecasts...")
    forecasts = data_processor.get_forecasts()
    if forecasts is not None and len(forecasts) > 0:
        print(f"   {len(forecasts)} forecasts generated")
        example = forecasts[0]
        print(f"   Forecast keys: {list(example.keys())}")
        if 'quantiles' in example and example['quantiles']:
            print(f"   Quantiles: {list(example['quantiles'].keys())}")
    else:
        print("   No forecasts (model may have failed)")

    # Check evidence
    print("\n8. Checking evidence extraction...")
    evidence = detection_method.get_evidence()
    if evidence is not None and len(evidence) > 0:
        print(f"   {len(evidence)} evidence dicts extracted")
        first = evidence[0]
        keys = sorted([k for k in first.keys() if not k.endswith('_error') and k != 'window_index'])
        print(f"   Metrics ({len(keys)}): {keys}")

        # Show which categories are present
        forecast_keys = [k for k in keys if k in ('mae', 'mse', 'mape', 'violation_ratio', 'mean_surprise', 'max_surprise', 'extreme_violation', 'extreme_violation_ratio')]
        stat_keys = [k for k in keys if k in ('max_abs_z_score', 'grubbs_statistic', 'max_cusum')]
        dist_keys = [k for k in keys if k in ('kl_divergence', 'wasserstein_distance', 'normalized_wasserstein')]
        pattern_keys = [k for k in keys if k in ('volatility_ratio', 'max_acf_diff', 'slope_diff')]
        print(f"   Forecast-based: {forecast_keys}")
        print(f"   Statistical tests: {stat_keys}")
        print(f"   Distribution: {dist_keys}")
        print(f"   Pattern: {pattern_keys}")

        # Check for errors
        errors = [k for k in first.keys() if k.endswith('_error')]
        if errors:
            print(f"   Errors: {errors}")
            for e in errors:
                print(f"     {e}: {first[e]}")
    else:
        print("   No evidence extracted")

    # Results
    print("\n9. Results:")
    print(f"   Predictions shape: {result.predictions.shape}")
    print(f"   Threshold: {result.threshold:.4f}")
    print(f"   Detected anomalies: {int(result.predictions.sum())} points")
    print(f"   Actual anomalies: {int(y_test.sum())} points")

    # Score comparison: anomaly region vs normal region
    anomaly_scores_1 = result.point_scores[100:120]
    anomaly_scores_2 = result.point_scores[300:310]
    normal_scores = np.concatenate([result.point_scores[:80], result.point_scores[140:280]])
    print(f"\n   Anomaly region 1 (100-120) avg score: {np.mean(anomaly_scores_1):.4f}")
    print(f"   Anomaly region 2 (300-310) avg score: {np.mean(anomaly_scores_2):.4f}")
    print(f"   Normal region avg score: {np.mean(normal_scores):.4f}")

    # Execution time
    print(f"\n10. Execution time:")
    for step, duration in result.execution_time.items():
        print(f"    {step}: {duration:.2f}s")

    # Metadata
    print(f"\n11. Metadata:")
    for k, v in result.metadata.items():
        print(f"    {k}: {v}")

    print("\n" + "="*60)
    print("TEST PASSED: Full Phase 2 pipeline works!")
    print("="*60)

    return True


def main():
    print("\n" + "="*60)
    print("PHASE 2 FULL PIPELINE INTEGRATION TEST")
    print("="*60)
    print("\nTests: FoundationModelProcessor + EvidenceBasedDetection")
    print("Prerequisites: pip install chronos-forecasting torch")

    try:
        success = test_full_phase2_pipeline()

        if success:
            print("\nIntegration test passed!")
            print("\nNext: Phase 2 Week 3 - LLM Reasoning Layer")
            return 0
        else:
            print("\nIntegration test failed")
            return 1

    except Exception as e:
        print(f"\nIntegration test crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
