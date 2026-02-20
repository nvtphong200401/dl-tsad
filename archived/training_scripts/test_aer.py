#!/usr/bin/env python
"""Test AER pipeline on small sample"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config_factory import load_config, build_pipeline_from_config
from src.data.anomllm_loader import load_anomllm_category
from src.evaluation.evaluator import Evaluator

def main():
    print("="*70)
    print("Testing AER Pipeline")
    print("="*70)

    # Load point category
    print("\nLoading 'point' category...")
    dataset = load_anomllm_category('point')
    print(f"Loaded: {len(dataset.X_test)} test samples")

    # Sample for faster testing
    X_train_sample = dataset.X_train[:5000]
    y_train_sample = dataset.y_train[:5000]
    X_val_sample = dataset.X_val[:2000]
    y_val_sample = dataset.y_val[:2000]
    X_test_sample = dataset.X_test[:2000]
    y_test_sample = dataset.y_test[:2000]

    print(f"\nUsing {len(X_train_sample)} train, {len(X_test_sample)} test samples")

    # Load AER config
    print("\nLoading AER config...")
    config = load_config("configs/pipelines/aer_pipeline.yaml")

    # Build pipeline
    print("Building AER pipeline...")
    pipeline = build_pipeline_from_config(config)

    # Train
    print("\n" + "-"*70)
    print("TRAINING")
    print("-"*70)
    pipeline.fit(X_train_sample, y_train_sample)

    # Validate
    print("\n" + "-"*70)
    print("VALIDATION")
    print("-"*70)
    val_result = pipeline.predict(X_val_sample, y_val_sample)

    evaluator = Evaluator()
    val_eval = evaluator.evaluate(y_val_sample, val_result.predictions, val_result.point_scores)
    print(f"Validation F1: {val_eval.f1:.3f}, PA-F1: {val_eval.pa_f1:.3f}")

    # Test
    print("\n" + "-"*70)
    print("TESTING")
    print("-"*70)
    test_result = pipeline.predict(X_test_sample, y_test_sample)

    # Evaluate
    eval_result = evaluator.evaluate(y_test_sample, test_result.predictions, test_result.point_scores)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"F1 Score:    {eval_result.f1:.3f}")
    print(f"Precision:   {eval_result.precision:.3f}")
    print(f"Recall:      {eval_result.recall:.3f}")
    print(f"PA-F1 Score: {eval_result.pa_f1:.3f}")
    print(f"Detected:    {test_result.predictions.sum()}/{y_test_sample.sum()}")
    print("="*70)

    print("\nAER test complete!")

if __name__ == "__main__":
    main()
