#!/usr/bin/env python
"""Comprehensive Benchmark: Phase 1 vs Phase 2 on AnomLLM datasets

Runs each (config, category) pair in a separate thread for maximum parallelism.
E.g., 4 configs x 4 categories = 16 threads.

Usage:
    # Run default configs on core 4 categories (16 threads)
    python experiments/run_benchmark.py

    # Limit threads (e.g., for GPU configs to avoid OOM)
    python experiments/run_benchmark.py --workers 4

    # All 8 categories
    python experiments/run_benchmark.py --all-categories

    # Sequential (for debugging)
    python experiments/run_benchmark.py --workers 1
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def run_one_job(config_name, category, data_cache, num_train, project_root):
    """Run one (config, category) pair. Each call runs in its own thread."""
    evaluator = Evaluator()
    config_path = os.path.join(project_root, f"configs/pipelines/{config_name}.yaml")
    label = f"{config_name:30s} | {category:10s}"

    if not os.path.exists(config_path):
        return {'error': f"Config not found: {config_path}", 'label': label}

    if category not in data_cache:
        return {'error': f"No data for {category}", 'label': label}

    train_series, eval_series = data_cache[category]

    try:
        # Select few-shot training series
        rng = np.random.RandomState(42)
        n = min(num_train, len(train_series))
        train_idx = rng.choice(len(train_series), n, replace=False)
        X_train = np.vstack([train_series[i][0] for i in train_idx])

        # Build and fit pipeline (each thread gets its own instance)
        config = load_config(config_path)
        pipeline = build_pipeline_from_config(config)
        pipeline.fit(X_train)

        # Evaluate per-series
        all_y_true, all_y_pred, all_scores = [], [], []
        total_time = 0
        n_ok = 0

        print(f"\n{label} | Evaluating on {len(eval_series)} series...")

        for series, labels in eval_series:
            try:
                t0 = time.time()
                result = pipeline.predict(series, labels)
                total_time += time.time() - t0
                all_y_true.append(labels)
                all_y_pred.append(result.predictions)
                all_scores.append(result.point_scores)
                n_ok += 1
                print(f"  {label} | Series {n_ok}/{len(eval_series)} evaluated ({len(labels)} points, {labels.sum()} anomalies)")
            except Exception:
                continue

        if n_ok == 0:
            return {'error': f"All {len(eval_series)} series failed", 'label': label}

        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        scores = np.concatenate(all_scores)
        print(f"  Evaluated {n_ok}/{len(eval_series)} series successfully ({len(y_true)} points)")
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
            'label': label,
        }

    except Exception as e:
        import traceback
        return {'error': traceback.format_exc(), 'label': label}


def main():
    parser = argparse.ArgumentParser(description='Benchmark Phase 1 vs Phase 2 on AnomLLM')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS)
    parser.add_argument('--categories', nargs='+', default=CORE_CATEGORIES)
    parser.add_argument('--all-categories', action='store_true')
    parser.add_argument('--num-train', type=int, default=5, help='Training series (few-shot)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Max parallel threads (default: all jobs)')
    parser.add_argument('--data-path', default='src/data/synthetic')
    parser.add_argument('--output', default='src/results/synthetic')
    args = parser.parse_args()

    categories = get_all_categories() if args.all_categories else args.categories
    configs = args.configs
    n_jobs = len(configs) * len(categories)
    max_workers = args.workers or n_jobs

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    data_path = os.path.join(project_root, args.data_path)

    print("=" * 70)
    print("BENCHMARK: Phase 1 vs Phase 2 on AnomLLM")
    print("=" * 70)
    print(f"Configs:      {configs}")
    print(f"Categories:   {categories}")
    print(f"Jobs:         {n_jobs} ({len(configs)} configs x {len(categories)} categories)")
    print(f"Threads:      {max_workers}")
    print(f"Train:        {args.num_train} series (few-shot)")
    print(f"Project root: {project_root}")

    # Load all datasets upfront (shared read-only across threads)
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

    # Split jobs into CPU-only (safe to parallelize) and GPU (must run sequentially)
    GPU_CONFIGS = {'phase2_timesfm', 'phase2_ensemble', 'phase2_llm', 'phase2_evidence_all', 'phase2_evidence'}
    cpu_jobs = [(cfg, cat) for cfg in configs for cat in categories
                if cat in data_cache and cfg not in GPU_CONFIGS]
    gpu_jobs = [(cfg, cat) for cfg in configs for cat in categories
                if cat in data_cache and cfg in GPU_CONFIGS]

    all_jobs_count = len(cpu_jobs) + len(gpu_jobs)
    print(f"\n{'='*70}")
    print(f"Total jobs: {all_jobs_count} ({len(cpu_jobs)} CPU-parallel + {len(gpu_jobs)} GPU-sequential)")
    print("-" * 70)

    os.makedirs(args.output, exist_ok=True)
    all_results = []
    t_start = time.time()

    def collect_result(result):
        label = result.get('label', '?')
        if 'error' in result:
            err = result['error']
            if len(err) > 200:
                err = err[:200] + "..."
            print(f"  {label} | FAILED: {err}")
        else:
            all_results.append(result)
            cat = result['category']
            n_eval = len(data_cache[cat][1]) if cat in data_cache else '?'
            print(f"  {label} | F1={result['f1']:.3f}  PA-F1={result['pa_f1']:.3f}  "
                  f"P={result['precision']:.3f}  R={result['recall']:.3f}  "
                  f"({result['time_s']:.0f}s, {result['n_series']}/{n_eval} series)")

    # Phase 1: Run CPU jobs in parallel
    if cpu_jobs:
        cpu_workers = max_workers or len(cpu_jobs)
        print(f"\n--- CPU jobs ({len(cpu_jobs)}) with {cpu_workers} threads ---")
        with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
            futures = {}
            for cfg, cat in cpu_jobs:
                future = executor.submit(
                    run_one_job, cfg, cat, data_cache, args.num_train, project_root
                )
                futures[future] = (cfg, cat)
            for future in as_completed(futures):
                collect_result(future.result())

    # Phase 2: Run GPU jobs sequentially (avoid model download/GPU contention)
    if gpu_jobs:
        print(f"\n--- GPU jobs ({len(gpu_jobs)}) sequential ---")
        for cfg, cat in gpu_jobs:
            result = run_one_job(cfg, cat, data_cache, args.num_train, project_root)
            collect_result(result)

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
