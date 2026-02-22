#!/usr/bin/env python
"""Comprehensive Benchmark: Phase 1 vs Phase 2 on AnomLLM datasets

Processes each series INDEPENDENTLY (matching AnomLLM's per-series evaluation).
Each (config, category) pair runs in a separate process for speed.

Usage:
    # Run all configs on core 4 categories (parallel)
    python experiments/run_benchmark.py

    # Run on all 8 categories
    python experiments/run_benchmark.py --all-categories

    # Control parallelism
    python experiments/run_benchmark.py --workers 4

    # Sequential (for debugging)
    python experiments/run_benchmark.py --workers 1
"""

import sys
import os
import time
import json
import argparse
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.anomllm_loader import get_all_categories

DEFAULT_CONFIGS = [
    'baseline_f1optimal',
    'phase2_evidence_fast',
    'phase2_evidence_small_window',
]

CORE_CATEGORIES = ['point', 'range', 'trend', 'freq']


def run_one_job(config_name, category, base_path, num_train, project_root):
    """Run one (config, category) pair. Designed to run in a subprocess.

    All imports happen inside so each process is self-contained.
    Returns dict with results or None on failure.
    """
    import sys, os, time, traceback
    import numpy as np

    # Use absolute project_root passed from parent (not __file__ which may fail in subprocess)
    sys.path.insert(0, project_root)
    os.chdir(project_root)

    try:
        from src.utils.config_factory import load_config, build_pipeline_from_config
        from src.data.anomllm_loader import load_anomllm_series
        from src.evaluation.evaluator import Evaluator
    except Exception as e:
        return {'error': f"Import failed: {traceback.format_exc()}", 'config': config_name, 'category': category}

    config_path = os.path.join(project_root, f"configs/pipelines/{config_name}.yaml")
    if not os.path.exists(config_path):
        return {'error': f"Config not found: {config_path}", 'config': config_name, 'category': category}

    data_path = os.path.join(project_root, base_path)

    # Load data
    try:
        train_series = load_anomllm_series(category, split='train', base_path=data_path)
        eval_series = load_anomllm_series(category, split='eval', base_path=data_path)
    except Exception as e:
        return {'error': f"Data load failed: {traceback.format_exc()}", 'config': config_name, 'category': category}

    # Select few-shot training series
    np.random.seed(42)
    n = min(num_train, len(train_series))
    train_idx = np.random.choice(len(train_series), n, replace=False)
    X_train = np.vstack([train_series[i][0] for i in train_idx])

    # Build and fit pipeline
    config = load_config(config_path)
    try:
        pipeline = build_pipeline_from_config(config)
        pipeline.fit(X_train)
    except Exception as e:
        return {'error': f"Fit failed: {traceback.format_exc()}", 'config': config_name, 'category': category}

    # Evaluate per-series
    evaluator = Evaluator()
    all_y_true, all_y_pred, all_scores = [], [], []
    total_time = 0
    n_ok = 0
    first_error = None

    for series, labels in eval_series:
        try:
            t0 = time.time()
            result = pipeline.predict(series, labels)
            total_time += time.time() - t0
            all_y_true.append(labels)
            all_y_pred.append(result.predictions)
            all_scores.append(result.point_scores)
            n_ok += 1
        except Exception as e:
            if first_error is None:
                first_error = traceback.format_exc()
            continue

    if n_ok == 0:
        return {'error': f'All series failed. First error: {first_error}', 'config': config_name, 'category': category}

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    scores = np.concatenate(all_scores)
    ev = evaluator.evaluate(y_true=y_true, y_pred=y_pred, scores=scores)

    return {
        'config': config_name,
        'category': category,
        'f1': ev.f1,
        'precision': ev.precision,
        'recall': ev.recall,
        'pa_f1': ev.pa_f1,
        'detected': int(y_pred.sum()),
        'total_anomalies': int(y_true.sum()),
        'n_series': n_ok,
        'total_points': len(y_true),
        'num_train_series': n,
        'time_s': total_time,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark Phase 1 vs Phase 2 on AnomLLM')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS)
    parser.add_argument('--categories', nargs='+', default=CORE_CATEGORIES)
    parser.add_argument('--all-categories', action='store_true')
    parser.add_argument('--num-train', type=int, default=5, help='Training series (few-shot)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: num categories)')
    parser.add_argument('--data-path', default='src/data/synthetic')
    parser.add_argument('--output', default='src/results/synthetic')
    args = parser.parse_args()

    categories = get_all_categories() if args.all_categories else args.categories
    configs = args.configs
    max_workers = args.workers or len(categories)

    print("=" * 70)
    print("BENCHMARK: Phase 1 vs Phase 2 on AnomLLM (parallel per-series)")
    print("=" * 70)
    print(f"Configs:    {configs}")
    print(f"Categories: {categories}")
    print(f"Workers:    {max_workers}")
    print(f"Train:      {args.num_train} series (few-shot)")

    # Resolve project root as absolute path (critical for subprocesses)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"Project root: {project_root}")

    os.makedirs(args.output, exist_ok=True)

    # Build all jobs: (config, category) pairs
    jobs = [(cfg, cat) for cfg in configs for cat in categories]
    print(f"\nTotal jobs: {len(jobs)} ({len(configs)} configs x {len(categories)} categories)")

    # Run in parallel
    t_start = time.time()
    all_results = []

    print(f"\nRunning {len(jobs)} jobs with {max_workers} workers...")
    print("-" * 70)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for cfg, cat in jobs:
            future = executor.submit(run_one_job, cfg, cat, args.data_path, args.num_train, project_root)
            futures[future] = (cfg, cat)

        for future in as_completed(futures):
            cfg, cat = futures[future]
            try:
                result = future.result()
                if result and 'error' not in result:
                    all_results.append(result)
                    print(f"  {cfg:30s} | {cat:10s} | "
                          f"F1={result['f1']:.3f}  PA-F1={result['pa_f1']:.3f}  "
                          f"P={result['precision']:.3f}  R={result['recall']:.3f}  "
                          f"({result['time_s']:.0f}s)")
                else:
                    err = result.get('error', 'Unknown') if result else 'No result'
                    print(f"  {cfg:30s} | {cat:10s} | FAILED: {err}")
            except Exception as e:
                print(f"  {cfg:30s} | {cat:10s} | CRASHED: {e}")

    wall_time = time.time() - t_start
    print(f"\nWall time: {wall_time:.0f}s")

    if not all_results:
        print("\nNo results!")
        return 1

    df = pd.DataFrame(all_results)

    # F1 comparison
    print(f"\n{'='*70}")
    print("F1 SCORE BY CATEGORY")
    print(f"{'='*70}")
    pivot = df.pivot_table(index='category', columns='config', values='f1')
    print(pivot.to_string(float_format='{:.3f}'.format))

    # PA-F1 comparison
    print(f"\n{'='*70}")
    print("PA-F1 SCORE BY CATEGORY")
    print(f"{'='*70}")
    pivot_pa = df.pivot_table(index='category', columns='config', values='pa_f1')
    print(pivot_pa.to_string(float_format='{:.3f}'.format))

    # Summary
    print(f"\n{'='*70}")
    print("AVERAGE ACROSS CATEGORIES")
    print(f"{'='*70}")
    summary = df.groupby('config').agg({
        'f1': 'mean', 'precision': 'mean', 'recall': 'mean',
        'pa_f1': 'mean', 'time_s': 'sum',
    }).round(3)
    summary.columns = ['Avg F1', 'Avg Prec', 'Avg Recall', 'Avg PA-F1', 'Total Time(s)']
    print(summary.to_string())

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output, f"benchmark_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    summary_json = {
        'timestamp': timestamp,
        'wall_time_s': wall_time,
        'configs': configs,
        'categories': categories,
        'num_train_series': args.num_train,
        'workers': max_workers,
        'per_config': {
            cfg: {
                'avg_f1': float(df[df['config'] == cfg]['f1'].mean()),
                'avg_pa_f1': float(df[df['config'] == cfg]['pa_f1'].mean()),
            }
            for cfg in configs if cfg in df['config'].values
        }
    }
    json_path = os.path.join(args.output, f"benchmark_summary_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f"[SAVED] {json_path}")

    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE (wall time: {wall_time:.0f}s)")
    print(f"{'='*70}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
