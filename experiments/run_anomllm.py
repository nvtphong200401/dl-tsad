#!/usr/bin/env python
"""Run experiments on AnomLLM synthetic dataset"""

import sys
import os
import json
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config_factory import load_config, build_pipeline_from_config
from src.data.anomllm_loader import load_all_anomllm_categories, get_all_categories
from src.evaluation.evaluator import Evaluator


def run_anomllm_experiments(config_path: str = "configs/pipelines/baseline_f1optimal.yaml"):
    """Run experiments on all AnomLLM categories

    Args:
        config_path: Path to pipeline configuration file
    """

    print("="*70)
    print("Best TSAD - AnomLLM Experiment Runner")
    print("="*70)

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

    for category, dataset in datasets.items():
        print("\n" + "="*70)
        print(f"CATEGORY: {category.upper()}")
        print("="*70)

        print(f"\nDataset info:")
        print(f"  Train size: {len(dataset.X_train)} ({dataset.metadata['train_anomalies']} anomalies)")
        print(f"  Val size:   {len(dataset.X_val)} ({dataset.metadata['val_anomalies']} anomalies)")
        print(f"  Test size:  {len(dataset.X_test)} ({dataset.metadata['test_anomalies']} anomalies)")
        print(f"  Dimensions: {dataset.X_train.shape[1]}")

        # Build pipeline
        print(f"\nBuilding pipeline...")
        pipeline = build_pipeline_from_config(config)

        # Train
        print("\nTraining...")
        try:
            pipeline.fit(dataset.X_train, dataset.y_train)
        except Exception as e:
            print(f"ERROR during training: {e}")
            continue

        # Validate (for threshold tuning)
        print("\nValidating (tuning threshold)...")
        try:
            val_result = pipeline.predict(dataset.X_val, dataset.y_val)
            print(f"  Threshold determined: {val_result.threshold:.4f}")

            val_eval = evaluator.evaluate(
                y_true=dataset.y_val,
                y_pred=val_result.predictions,
                scores=val_result.point_scores
            )
            print(f"  Validation F1: {val_eval.f1:.3f} (P: {val_eval.precision:.3f}, R: {val_eval.recall:.3f})")
        except Exception as e:
            print(f"ERROR during validation: {e}")
            continue

        # Test
        print("\nTesting...")
        try:
            test_result = pipeline.predict(dataset.X_test, dataset.y_test)

            # Evaluate
            eval_result = evaluator.evaluate(
                y_true=dataset.y_test,
                y_pred=test_result.predictions,
                scores=test_result.point_scores
            )

            # Print results
            print("\n" + "-"*70)
            print("RESULTS")
            print("-"*70)
            print(f"F1 Score:          {eval_result.f1:.3f}")
            print(f"Precision:         {eval_result.precision:.3f}")
            print(f"Recall:            {eval_result.recall:.3f}")
            print(f"PA-F1 Score:       {eval_result.pa_f1:.3f}")
            print(f"Detected:          {test_result.predictions.sum()}/{dataset.y_test.sum()}")
            print(f"Inference time:    {test_result.execution_time['total']:.4f}s")
            print("-"*70)

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
                'total_anomalies': int(dataset.y_test.sum()),
                'test_size': len(dataset.X_test),
                'inference_time': test_result.execution_time['total'],
                'train_time': pipeline.execution_time.get('step1_fit', 0) + pipeline.execution_time.get('step2_fit', 0)
            })

        except Exception as e:
            print(f"ERROR during testing: {e}")
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
        csv_filename = f"src/results/synthetic/results_{pipeline_name}_{timestamp}.csv"
        df_results.to_csv(csv_filename, index=False)
        print(f"\n[SAVED] Results saved to: {csv_filename}")

        # Save summary statistics
        summary = {
            'pipeline': pipeline_name,
            'config': config_path,
            'timestamp': timestamp,
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

        json_filename = f"src/results/synthetic/summary_{pipeline_name}_{timestamp}.json"
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

    parser = argparse.ArgumentParser(description='Run AnomLLM experiments')
    parser.add_argument('--config', type=str, default='configs/pipelines/baseline_f1optimal.yaml',
                      help='Path to pipeline config file')
    args = parser.parse_args()

    run_anomllm_experiments(args.config)


if __name__ == "__main__":
    main()
