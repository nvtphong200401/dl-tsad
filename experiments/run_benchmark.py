#!/usr/bin/env python
"""Comprehensive Benchmark: Phase 1 vs Phase 2 on AnomLLM datasets

Runs each config in a separate thread (1 thread per config).
Within each thread, categories run sequentially.

Usage:
    # Run default configs on core 4 categories
    python experiments/run_benchmark.py

    # All 8 categories
    python experiments/run_benchmark.py --all-categories

    # Specific configs
    python experiments/run_benchmark.py --configs baseline_f1optimal phase2_timesfm

    # Sequential (for debugging)
    python experiments/run_benchmark.py --sequential
"""

import sys
import os
import time
import json
import argparse
import threading
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config_factory import load_config, build_pipeline_from_config
from src.data.anomllm_loader import load_anomllm_series, get_all_categories
from src.evaluation.evaluator import Evaluator

DEFAULT_CONFIGS = [
    'baseline_f1optimal',
    'phase2_evidence_fast',
    'phase2_evidence_small_window',
]

CORE_CATEGORIES = ['point', 'range', 'trend', 'freq']

# Thread-safe results collection
_results_lock = threading.Lock()
_all_results = []


def run_config_thread(config_name, categories, data_cache, num_train, project_root):
    """Run one config across all categories. Runs in its own thread."""
    evaluator = Evaluator()
    config_path = os.path.join(project_root, f"configs/pipelines/{config_name}.yaml")

    if not os.path.exists(config_path):
        print(f"  [{config_name}] Config not found: {config_path}")
        return

    config = load_config(config_path)

    for category in categories:
        if category not in data_cache:
            continue

        train_series, eval_series = data_cache[category]
        label = f"{config_name:30s} | {category:10s}"

        try:
            # Select few-shot training series
            np.random.seed(42)
            n = min(num_train, len(train_series))
            train_idx = np.random.choice(len(train_series), n, replace=False)
            X_train = np.vstack([train_series[i][0] for i in train_idx])

            # Build and fit pipeline
            pipeline = build_pipeline_from_config(config)
            pipeline.fit(X_train)

            # Evaluate per-series
            all_y_true, all_y_pred, all_scores = [], [], []
            total_time = 0
            n_ok = 0

            for series, labels in eval_series:
                try:
                    t0 = time.time()
                    result = pipeline.predict(series, labels)
                    total_time += time.time() - t0
                    all_y_true.append(labels)
                    all_y_pred.append(result.predictions)
                    all_scores.append(result.point_scores)
                    n_ok += 1
                except Exception:
                    continue

            if n_ok == 0:
                print(f"  {label} | FAILED: all {len(eval_series)} series failed")
                continue

            y_true = np.concatenate(all_y_true)
            y_pred = np.concatenate(all_y_pred)
            scores = np.concatenate(all_scores)
            ev = evaluator.evaluate(y_true=y_true, y_pred=y_pred, scores=scores)

            row = {
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

            with _results_lock:
                _all_results.append(row)

            print(f"  {label} | F1={ev.f1:.3f}  PA-F1={ev.pa_f1:.3f}  "
                  f"P={ev.precision:.3f}  R={ev.recall:.3f}  "
                  f"({total_time:.0f}s, {n_ok}/{len(eval_series)} series)")

        except Exception as e:
            import traceback
            print(f"  {label} | FAILED: {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Benchmark Phase 1 vs Phase 2 on AnomLLM')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS)
    parser.add_argument('--categories', nargs='+', default=CORE_CATEGORIES)
    parser.add_argument('--all-categories', action='store_true')
    parser.add_argument('--num-train', type=int, default=5, help='Training series (few-shot)')
    parser.add_argument('--sequential', action='store_true', help='Run sequentially (no threads)')
    parser.add_argument('--data-path', default='src/data/synthetic')
    parser.add_argument('--output', default='src/results/synthetic')
    args = parser.parse_args()

    categories = get_all_categories() if args.all_categories else args.categories
    configs = args.configs

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    data_path = os.path.join(project_root, args.data_path)

    print("=" * 70)
    print("BENCHMARK: Phase 1 vs Phase 2 on AnomLLM")
    print("=" * 70)
    print(f"Configs:      {configs}")
    print(f"Categories:   {categories}")
    print(f"Parallel:     {'no (sequential)' if args.sequential else f'{len(configs)} threads (1 per config)'}")
    print(f"Train:        {args.num_train} series (few-shot)")
    print(f"Project root: {project_root}")

    # Load all datasets upfront (shared across threads, read-only)
    print(f"\nLoading datasets...")
    data_cache = {}
    for cat in categories:
        try:
            train = load_anomllm_series(cat, split='train', base_path=data_path)
            test = load_anomllm_series(cat, split='eval', base_path=data_path)
            data_cache[cat] = (train, test)
            total_pts = sum(len(s) for s, _ in test)
            n_anom = sum(int(l.sum()) for _, l in test)
            print(f"  {cat:15s}: {len(train)} train, {len(test)} eval ({total_pts} pts, {n_anom} anomalies)")
        except Exception as e:
            print(f"  {cat:15s}: FAILED - {e}")

    if not data_cache:
        print("\nNo datasets loaded!")
        return 1

    # Run benchmarks
    print(f"\n{'='*70}")
    print(f"Running {len(configs)} configs x {len(categories)} categories...")
    print("-" * 70)

    global _all_results
    _all_results = []
    t_start = time.time()

    os.makedirs(args.output, exist_ok=True)

    if args.sequential:
        for cfg in configs:
            run_config_thread(cfg, categories, data_cache, args.num_train, project_root)
    else:
        threads = []
        for cfg in configs:
            t = threading.Thread(
                target=run_config_thread,
                args=(cfg, categories, data_cache, args.num_train, project_root)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    wall_time = time.time() - t_start
    print(f"\nWall time: {wall_time:.0f}s")

    if not _all_results:
        print("\nNo results!")
        return 1

    df = pd.DataFrame(_all_results)

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
