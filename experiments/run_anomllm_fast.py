#!/usr/bin/env python
"""Run fast experiments on AnomLLM synthetic dataset (with sampling)"""

import sys
import os
import json
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config_factory import load_config, build_pipeline_from_config
from src.data.anomllm_loader import load_all_anomllm_categories
from src.evaluation.evaluator import Evaluator


def run_anomllm_experiments_fast(config_path: str = "configs/pipelines/baseline_f1optimal.yaml",
                                 train_samples: int = 50000,
                                 test_samples: int = 50000):
    """Run experiments on all AnomLLM categories with sampling for speed

    Args:
        config_path: Path to pipeline configuration file
        train_samples: Number of training samples to use (default: 50000)
        test_samples: Number of test samples to use (default: 50000)
    """

    print("="*70)
    print("Best TSAD - AnomLLM Fast Experiment Runner")
    print("="*70)
    print(f"Training samples: {train_samples}")
    print(f"Test samples: {test_samples}")

    # Create results directory
    os.makedirs("src/results/synthetic", exist_ok=True)

    # Load config
    print(f"\nLoading config from: {config_path}")
    config = load_config(config_path)
    pipeline_name = config['experiment']['name']
    print(f"Pipeline: {pipeline_name}")

    # Load all AnomLLM datasets
    print("\n" + "-"*70)
    print("Loading AnomLLM datasets...")
    print("-"*70)
    datasets = load_all_anomllm_categories()

    if not datasets:
        print("ERROR: No datasets loaded!")
        return

    print(f"\nSuccessfully loaded {len(datasets)} categories")

    # Run experiments on each category
    results = []
    evaluator = Evaluator()

    for idx, (category, dataset) in enumerate(datasets.items(), 1):
        print("\n" + "="*70)
        print(f"[{idx}/{len(datasets)}] CATEGORY: {category.upper()}")
        print("="*70)

        # Sample datasets for faster processing
        X_train_sample = dataset.X_train[:train_samples]
        y_train_sample = dataset.y_train[:train_samples]
        X_test_sample = dataset.X_test[:test_samples]
        y_test_sample = dataset.y_test[:test_samples]

        # Also sample validation set
        X_val_sample = dataset.X_val[:test_samples]
        y_val_sample = dataset.y_val[:test_samples]

        print(f"\nDataset info (sampled):")
        print(f"  Train size: {len(X_train_sample)}")
        print(f"  Val size:   {len(X_val_sample)} ({y_val_sample.sum()} anomalies)")
        print(f"  Test size:  {len(X_test_sample)} ({y_test_sample.sum()} anomalies)")
        print(f"  Dimensions: {X_train_sample.shape[1]}")

        # Build pipeline
        print(f"\nBuilding pipeline...")
        pipeline = build_pipeline_from_config(config)

        # Train
        print("\nTraining...")
        try:
            pipeline.fit(X_train_sample, y_train_sample)
        except Exception as e:
            print(f"ERROR during training: {e}")
            continue

        # Validate (for threshold tuning)
        print("Validating...")
        try:
            val_result = pipeline.predict(X_val_sample, y_val_sample)
            val_eval = evaluator.evaluate(
                y_true=y_val_sample,
                y_pred=val_result.predictions,
                scores=val_result.point_scores
            )
            print(f"  Threshold: {val_result.threshold:.4f}, Val F1: {val_eval.f1:.3f}")
        except Exception as e:
            print(f"ERROR during validation: {e}")
            continue

        # Test
        print("Testing...")
        try:
            test_result = pipeline.predict(X_test_sample, y_test_sample)

            # Evaluate
            eval_result = evaluator.evaluate(
                y_true=y_test_sample,
                y_pred=test_result.predictions,
                scores=test_result.point_scores
            )

            # Print results
            print(f"\nRESULTS: F1={eval_result.f1:.3f}, P={eval_result.precision:.3f}, "
                  f"R={eval_result.recall:.3f}, PA-F1={eval_result.pa_f1:.3f}")

            # Store results
            results.append({
                'category': category,
                'pipeline': pipeline_name,
                'f1': eval_result.f1,
                'precision': eval_result.precision,
                'recall': eval_result.recall,
                'pa_f1': eval_result.pa_f1,
                'threshold': test_result.threshold,
                'detected': int(test_result.predictions.sum()),
                'total_anomalies': int(y_test_sample.sum()),
                'test_size': len(X_test_sample),
                'train_size': len(X_train_sample),
                'inference_time': test_result.execution_time['total']
            })

        except Exception as e:
            print(f"ERROR during testing: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    if results:
        print("\n" + "="*70)
        print("SUMMARY OF ALL RESULTS")
        print("="*70)

        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))

        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"src/results/synthetic/results_{pipeline_name}_fast_{timestamp}.csv"
        df_results.to_csv(csv_filename, index=False)
        print(f"\n[SAVED] Results saved to: {csv_filename}")

        # Save summary statistics
        summary = {
            'pipeline': pipeline_name,
            'config': config_path,
            'timestamp': timestamp,
            'train_samples': train_samples,
            'test_samples': test_samples,
            'num_categories': len(results),
            'avg_f1': float(df_results['f1'].mean()),
            'avg_precision': float(df_results['precision'].mean()),
            'avg_recall': float(df_results['recall'].mean()),
            'avg_pa_f1': float(df_results['pa_f1'].mean()),
            'best_category': df_results.loc[df_results['f1'].idxmax(), 'category'],
            'best_f1': float(df_results['f1'].max()),
            'worst_category': df_results.loc[df_results['f1'].idxmin(), 'category'],
            'worst_f1': float(df_results['f1'].min()),
        }

        json_filename = f"src/results/synthetic/summary_{pipeline_name}_fast_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[SAVED] Summary saved to: {json_filename}")

        # Print summary statistics
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(f"Average F1:        {summary['avg_f1']:.3f}")
        print(f"Average Precision: {summary['avg_precision']:.3f}")
        print(f"Average Recall:    {summary['avg_recall']:.3f}")
        print(f"Average PA-F1:     {summary['avg_pa_f1']:.3f}")
        print(f"\nBest:  {summary['best_category']:15s} (F1: {summary['best_f1']:.3f})")
        print(f"Worst: {summary['worst_category']:15s} (F1: {summary['worst_f1']:.3f})")
        print("="*70)

    else:
        print("\nERROR: No results to save!")

    print("\n" + "="*70)
    print("Experiment complete!")
    print("="*70)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Run AnomLLM experiments (fast mode)')
    parser.add_argument('--config', type=str, default='configs/pipelines/baseline_f1optimal.yaml',
                      help='Path to pipeline config file')
    parser.add_argument('--train-samples', type=int, default=50000,
                      help='Number of training samples to use')
    parser.add_argument('--test-samples', type=int, default=50000,
                      help='Number of test samples to use')
    args = parser.parse_args()

    run_anomllm_experiments_fast(args.config, args.train_samples, args.test_samples)


if __name__ == "__main__":
    main()
