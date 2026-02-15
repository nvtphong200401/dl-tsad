#!/usr/bin/env python
"""Run baseline experiment with synthetic data"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config_factory import load_config, build_pipeline_from_config
from src.data.loader import create_synthetic_dataset
from src.evaluation.evaluator import Evaluator


def main():
    """Run baseline experiment"""

    print("="*60)
    print("Best TSAD - Baseline Experiment")
    print("="*60)

    # Load config
    config_path = "configs/pipelines/baseline.yaml"
    print(f"\nLoading config from: {config_path}")
    config = load_config(config_path)

    # Create synthetic dataset
    print("\nCreating synthetic dataset...")
    dataset = create_synthetic_dataset(
        n_samples=2000,
        n_dims=5,
        anomaly_ratio=0.05,
        random_seed=42
    )

    print(f"  Train size: {len(dataset.X_train)} (all normal)")
    print(f"  Val size:   {len(dataset.X_val)} ({dataset.y_val.sum()} anomalies)")
    print(f"  Test size:  {len(dataset.X_test)} ({dataset.y_test.sum()} anomalies)")
    print(f"  Dimensions: {dataset.X_train.shape[1]}")

    # Build pipeline
    print(f"\nBuilding pipeline: {config['experiment']['name']}")
    pipeline = build_pipeline_from_config(config)

    # Train
    print("\n" + "-"*60)
    print("TRAINING")
    print("-"*60)
    pipeline.fit(dataset.X_train, dataset.y_train)

    # Predict on validation (for threshold tuning if using F1Optimal)
    print("\n" + "-"*60)
    print("VALIDATION")
    print("-"*60)
    print("Running inference on validation set...")
    val_result = pipeline.predict(dataset.X_val, dataset.y_val)
    print(f"  Threshold determined: {val_result.threshold:.4f}")

    # Predict on test
    print("\n" + "-"*60)
    print("TESTING")
    print("-"*60)
    print("Running inference on test set...")
    test_result = pipeline.predict(dataset.X_test, dataset.y_test)

    # Evaluate
    print("\nEvaluating results...")
    evaluator = Evaluator()
    eval_result = evaluator.evaluate(
        y_true=dataset.y_test,
        y_pred=test_result.predictions,
        scores=test_result.point_scores
    )

    # Print results
    print("\n" + "="*60)
    print(f"RESULTS: {config['experiment']['name']}")
    print("="*60)
    print(f"F1 Score:          {eval_result.f1:.3f}")
    print(f"Precision:         {eval_result.precision:.3f}")
    print(f"Recall:            {eval_result.recall:.3f}")
    print(f"PA-F1 Score:       {eval_result.pa_f1:.3f}")
    print("="*60)

    # Print pipeline timing breakdown
    print(f"\nPipeline Timing Breakdown:")
    for step, time in test_result.execution_time.items():
        print(f"  {step:20s}: {time:.4f}s")

    print(f"\nDetected {test_result.predictions.sum()}/{dataset.y_test.sum()} anomalies")
    print(f"Threshold used: {test_result.threshold:.4f}")

    print("\n" + "="*60)
    print("Experiment complete!")
    print("="*60)


if __name__ == "__main__":
    main()
