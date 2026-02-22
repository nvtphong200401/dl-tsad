#!/usr/bin/env python
"""Test Phase 2 full pipeline: Foundation Models + Evidence + LLM Scoring

This script tests two pipeline configurations:
  Mode 1 (Heuristic): Chronos → Evidence → MaxPooling → Threshold
  Mode 2 (LLM):       Chronos → Evidence → LLMReasoningScoring → Threshold

Usage:
    # Heuristic scoring (no API key needed)
    python experiments/test_phase2_pipeline_integration.py

    # LLM scoring (requires Azure OpenAI API key in .env)
    python experiments/test_phase2_pipeline_integration.py --llm
"""

import sys
import argparse
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
    """Create synthetic time series with anomalies"""
    t = np.linspace(0, 10*np.pi, length)
    X = np.sin(t) + 0.1 * np.random.randn(length)
    y = np.zeros(length, dtype=int)

    if anomaly_positions:
        for start, end in anomaly_positions:
            X[start:end] += 5.0
            y[start:end] = 1

    return X, y


def run_pipeline(scoring_method, scoring_name, X_train, y_train, X_test, y_test):
    """Run a pipeline with given scoring method and print results."""

    print(f"\n{'='*60}")
    print(f"Pipeline: Chronos + Evidence + {scoring_name}")
    print(f"{'='*60}")

    # Step 1: Foundation Model Processor
    # Use window_size=30 for better anomaly sensitivity (anomalies are 10-20 points)
    window_config = WindowConfig(window_size=30, stride=1)
    data_processor = FoundationModelProcessor(
        window_config=window_config,
        forecast_horizon=30,
        models=['chronos'],
        chronos_model="amazon/chronos-t5-tiny",
        num_samples=20
    )

    # Step 2: Evidence-based detection
    # Disable forecast_based: the forecast predicts future points, not the current window,
    # so comparing them adds noise. Use statistical + distribution + pattern evidence.
    detection_method = EvidenceBasedDetection(
        enabled_categories=['statistical_tests', 'distribution_based', 'pattern_based']
    )

    # Step 4: Post-processing
    # Use 90th percentile: anomalies are ~4-6% of windows, so 95th is too tight
    post_processor = PostProcessor(
        threshold_method=PercentileThreshold(percentile=90.0),
        min_anomaly_length=3,
        merge_gap=5
    )

    # Build pipeline
    pipeline = AnomalyDetectionPipeline(
        data_processor=data_processor,
        detection_method=detection_method,
        scoring_method=scoring_method,
        post_processor=post_processor
    )

    # Fit
    print("\nFitting pipeline...")
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"  Fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Predict
    print("Running prediction...")
    try:
        result = pipeline.predict(X_test, y_test)
    except Exception as e:
        print(f"  Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Check evidence
    evidence = detection_method.get_evidence()
    if evidence and len(evidence) > 0:
        first = evidence[0]
        metric_keys = [k for k in first.keys() if not k.endswith('_error') and k != 'window_index']
        print(f"\n  Evidence: {len(evidence)} windows, {len(metric_keys)} metrics each")
        errors = [k for k in first.keys() if k.endswith('_error')]
        if errors:
            for e in errors:
                print(f"    Warning: {e}: {first[e]}")

    # Score comparison
    anomaly_scores_1 = result.point_scores[100:120]
    anomaly_scores_2 = result.point_scores[300:310]
    normal_scores = np.concatenate([result.point_scores[:80], result.point_scores[140:280]])

    print(f"\n  Results:")
    print(f"    Detected: {int(result.predictions.sum())} points")
    print(f"    Actual:   {int(y_test.sum())} points")
    print(f"    Threshold: {result.threshold:.4f}")
    print(f"\n  Score comparison:")
    print(f"    Anomaly region 1 (100-120): {np.mean(anomaly_scores_1):.4f}")
    print(f"    Anomaly region 2 (300-310): {np.mean(anomaly_scores_2):.4f}")
    print(f"    Normal region avg:          {np.mean(normal_scores):.4f}")
    ratio = np.mean(anomaly_scores_1) / (np.mean(normal_scores) + 1e-10)
    print(f"    Anomaly/Normal ratio:       {ratio:.2f}x")

    # Execution time
    print(f"\n  Execution time:")
    for step, duration in result.execution_time.items():
        print(f"    {step}: {duration:.2f}s")

    # LLM-specific info
    if hasattr(scoring_method, 'get_call_count'):
        calls = scoring_method.get_call_count()
        if calls > 0:
            print(f"\n  LLM API calls: {calls}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Pipeline Integration Test")
    parser.add_argument('--llm', action='store_true', help='Enable LLM scoring (requires API key)')
    parser.add_argument('--backend', default='azure_openai', choices=['azure_openai', 'gemini', 'claude'],
                        help='LLM backend (default: azure_openai)')
    parser.add_argument('--batch-size', type=int, default=10, help='Windows per LLM call (default: 10)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("PHASE 2 PIPELINE INTEGRATION TEST")
    print("="*60)

    # Create synthetic data
    np.random.seed(42)
    X_train, y_train = create_synthetic_data(length=1000, anomaly_positions=None)
    X_test, y_test = create_synthetic_data(length=500, anomaly_positions=[(100, 120), (300, 310)])
    print(f"\nData: Train={len(X_train)} (normal), Test={len(X_test)} ({y_test.sum()} anomalous)")

    results = {}

    # Mode 1: Heuristic scoring (always run)
    results['heuristic'] = run_pipeline(
        MaxPoolingScoring(), "MaxPoolingScoring (heuristic)",
        X_train, y_train, X_test, y_test
    )

    # Mode 2: LLM scoring (if --llm flag)
    if args.llm:
        try:
            from pipeline.step3_scoring import LLMReasoningScoring
            llm_scoring = LLMReasoningScoring(
                backend_type=args.backend,
                batch_size=args.batch_size
            )
            results['llm'] = run_pipeline(
                llm_scoring, f"LLMReasoningScoring ({args.backend})",
                X_train, y_train, X_test, y_test
            )
        except Exception as e:
            print(f"\n  LLM scoring setup failed: {e}")
            print("  Check your .env file has valid API credentials.")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, result in results.items():
        if result is not None:
            detected = int(result.predictions.sum())
            actual = int(y_test.sum())
            total_time = result.execution_time.get('total', 0)
            print(f"  {name:12s}: detected {detected}/{actual} anomalies, time={total_time:.1f}s")
        else:
            print(f"  {name:12s}: FAILED")

    success = all(r is not None for r in results.values())
    print(f"\n{'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'}")
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
