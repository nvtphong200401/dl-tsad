#!/usr/bin/env python
"""Test run on a single AnomLLM category"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config_factory import load_config, build_pipeline_from_config
from src.data.anomllm_loader import load_anomllm_category
from src.evaluation.evaluator import Evaluator

def main():
    # Load single category
    print("Loading 'point' category...")
    dataset = load_anomllm_category('point')
    print(f"Loaded: {len(dataset.X_test)} test samples")

    # Load config
    print("\nLoading config...")
    config = load_config("configs/pipelines/baseline_f1optimal.yaml")

    # Build pipeline
    print("Building pipeline...")
    pipeline = build_pipeline_from_config(config)

    # Train on subset (first 10000 samples)
    print("\nTraining on subset...")
    X_train_small = dataset.X_train[:10000]
    y_train_small = dataset.y_train[:10000]
    pipeline.fit(X_train_small, y_train_small)
    print("Training complete!")

    # Test on subset
    print("\nTesting on subset...")
    X_test_small = dataset.X_test[:5000]
    y_test_small = dataset.y_test[:5000]
    result = pipeline.predict(X_test_small, y_test_small)

    # Evaluate
    evaluator = Evaluator()
    eval_result = evaluator.evaluate(y_test_small, result.predictions, result.point_scores)

    print(f"\nResults:")
    print(f"  F1: {eval_result.f1:.3f}")
    print(f"  Precision: {eval_result.precision:.3f}")
    print(f"  Recall: {eval_result.recall:.3f}")
    print(f"  PA-F1: {eval_result.pa_f1:.3f}")

if __name__ == "__main__":
    main()
