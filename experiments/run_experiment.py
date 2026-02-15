#!/usr/bin/env python
"""Run experiment with configurable config file"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config_factory import load_config, build_pipeline_from_config
from src.data.loader import create_synthetic_dataset
from src.evaluation.evaluator import Evaluator


def main():
    """Run experiment"""

    parser = argparse.ArgumentParser(description='Run anomaly detection experiment')
    parser.add_argument('--config', type=str, default='configs/pipelines/baseline.yaml',
                      help='Path to config file')
    parser.add_argument('--n-samples', type=int, default=2000,
                      help='Number of samples in synthetic dataset')
    parser.add_argument('--anomaly-ratio', type=float, default=0.05,
                      help='Anomaly ratio for synthetic dataset')
    args = parser.parse_args()

    print("="*60)
    print("Best TSAD - Experiment Runner")
    print("="*60)

    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)

    # Create synthetic dataset
    print("\nCreating synthetic dataset...")
    dataset = create_synthetic_dataset(
        n_samples=args.n_samples,
        n_dims=5,
        anomaly_ratio=args.anomaly_ratio,
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

    # Predict on validation (for threshold tuning)
    print("\n" + "-"*60)
    print("VALIDATION (Threshold Tuning)")
    print("-"*60)
    print("Running inference on validation set...")
    val_result = pipeline.predict(dataset.X_val, dataset.y_val)
    print(f"  Threshold determined: {val_result.threshold:.4f}")

    # Evaluate on validation
    evaluator = Evaluator()
    val_eval = evaluator.evaluate(
        y_true=dataset.y_val,
        y_pred=val_result.predictions,
        scores=val_result.point_scores
    )
    print(f"  Validation F1: {val_eval.f1:.3f} (Precision: {val_eval.precision:.3f}, Recall: {val_eval.recall:.3f})")

    # Predict on test
    print("\n" + "-"*60)
    print("TESTING")
    print("-"*60)
    print("Running inference on test set...")
    test_result = pipeline.predict(dataset.X_test, dataset.y_test)

    # Evaluate
    print("\nEvaluating results...")
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

    # Print anomaly statistics
    if eval_result.f1 > 0:
        print(f"\nTrue Positives:  {int(eval_result.recall * dataset.y_test.sum())}")
        print(f"False Positives: {int(test_result.predictions.sum() - eval_result.recall * dataset.y_test.sum())}")
        print(f"False Negatives: {int((1 - eval_result.recall) * dataset.y_test.sum())}")

    print("\n" + "="*60)
    print("Experiment complete!")
    print("="*60)


if __name__ == "__main__":
    main()
