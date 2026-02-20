# Phase 4: Evaluation & Optimization (Weeks 7-8)

## Overview

This final phase focuses on comprehensive evaluation, ablation studies, performance optimization, and production readiness. The goal is to validate the system against benchmarks, identify critical components, optimize for production, and prepare for deployment.

**Timeline**: 2 weeks
**Status**: 📋 Planned
**Prerequisites**: Phase 3 (Advanced Features) completed

---

## Objectives

1. ✅ Comprehensive benchmarking on standard datasets
2. ✅ Ablation studies to understand component contributions
3. ✅ Performance optimization (speed, memory, cost)
4. ✅ Comparison to baselines and SOTA methods
5. ✅ Production deployment preparation
6. ✅ Research paper / technical report writing

**Goal**: By end of Phase 4, achieve publishable results (F1 > 0.75), understand system behavior, and have production-ready deployment.

---

## Phase Breakdown

### Week 1: Comprehensive Evaluation

**Deliverable**: Complete evaluation results on multiple benchmark datasets

#### Tasks

**1.1 Benchmark Dataset Preparation**

**Datasets to Evaluate**:
- **SMD** (Server Machine Dataset): 38 machines, multivariate
- **MSL** (Mars Science Laboratory): NASA telemetry
- **SMAP** (Soil Moisture Active Passive): NASA telemetry
- **SWAT** (Secure Water Treatment): Industrial control
- **PSM** (Pooled Server Metrics): Server monitoring
- **Yahoo S5**: Web traffic anomalies
- **NAB** (Numenta Anomaly Benchmark): Real-world time series

**File**: `experiments/prepare_benchmarks.py`
```python
#!/usr/bin/env python
"""Prepare and validate benchmark datasets."""

from src.data.loader import (
    load_smd_dataset,
    load_msl_smap_dataset,
    load_swat_dataset,
    load_psm_dataset,
    load_yahoo_dataset,
    load_nab_dataset
)

def prepare_all_datasets():
    """Download and prepare all benchmark datasets."""

    datasets = {
        'smd': load_smd_dataset,
        'msl': lambda: load_msl_smap_dataset('MSL'),
        'smap': lambda: load_msl_smap_dataset('SMAP'),
        'swat': load_swat_dataset,
        'psm': load_psm_dataset,
        'yahoo_s5': load_yahoo_dataset,
        'nab': load_nab_dataset
    }

    prepared = {}
    for name, loader in datasets.items():
        print(f"Preparing {name}...")
        try:
            dataset = loader()
            prepared[name] = {
                'train_size': len(dataset.X_train),
                'test_size': len(dataset.X_test),
                'anomaly_ratio': dataset.y_test.sum() / len(dataset.y_test),
                'dimensions': dataset.X_train.shape[1] if dataset.X_train.ndim > 1 else 1
            }
            print(f"  ✓ {name}: {prepared[name]}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    return prepared

if __name__ == '__main__':
    prepare_all_datasets()
```

**1.2 Comprehensive Evaluation Script**

**File**: `experiments/comprehensive_evaluation.py`
```python
#!/usr/bin/env python
"""Run comprehensive evaluation on all benchmarks."""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from src.utils.config_factory import build_pipeline_from_config
from src.data.loader import load_dataset
from src.evaluation.evaluator import Evaluator

def evaluate_single_dataset(
    config_path: str,
    dataset_name: str,
    output_dir: Path
) -> Dict:
    """Evaluate pipeline on single dataset."""

    # Load configuration
    config = load_config(config_path)
    pipeline = build_pipeline_from_config(config)

    # Load dataset
    dataset = load_dataset(dataset_name)

    # Fit pipeline
    print(f"  Fitting on {len(dataset.X_train)} training samples...")
    pipeline.fit(dataset.X_train)

    # Predict
    print(f"  Predicting on {len(dataset.X_test)} test samples...")
    result = pipeline.predict(dataset.X_test, dataset.y_test)

    # Evaluate
    evaluator = Evaluator()
    metrics = evaluator.evaluate(
        y_true=dataset.y_test,
        y_pred=result['predictions'],
        scores=result.get('scores')
    )

    # Save detailed results
    output_file = output_dir / f"{dataset_name}_results.json"
    save_results(output_file, metrics, result, config)

    return metrics.to_dict()

def run_comprehensive_evaluation(
    config_paths: List[str],
    datasets: List[str],
    output_dir: Path
):
    """Run evaluation across all configs and datasets."""

    results = []

    for config_path in config_paths:
        config_name = Path(config_path).stem

        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Evaluating: {config_name} on {dataset_name}")
            print(f"{'='*60}")

            try:
                metrics = evaluate_single_dataset(
                    config_path=config_path,
                    dataset_name=dataset_name,
                    output_dir=output_dir / config_name
                )

                results.append({
                    'config': config_name,
                    'dataset': dataset_name,
                    **metrics
                })

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results.append({
                    'config': config_name,
                    'dataset': dataset_name,
                    'error': str(e)
                })

    # Save summary
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'evaluation_summary.csv', index=False)

    # Print summary table
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(df_results.pivot_table(
        index='config',
        columns='dataset',
        values='f1',
        aggfunc='mean'
    ))

    return df_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True,
                       help='List of config files')
    parser.add_argument('--datasets', nargs='+', required=True,
                       help='List of datasets')
    parser.add_argument('--output', default='results/comprehensive',
                       help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_comprehensive_evaluation(
        config_paths=args.configs,
        datasets=args.datasets,
        output_dir=output_dir
    )

if __name__ == '__main__':
    main()
```

**1.3 Baseline Comparisons**

**File**: `experiments/compare_baselines.py`
```python
#!/usr/bin/env python
"""Compare against baseline methods."""

from typing import List, Dict
import numpy as np
from src.data.loader import load_dataset
from src.evaluation.evaluator import Evaluator

def run_baseline_comparison(dataset_name: str) -> pd.DataFrame:
    """Compare Phase 3 against multiple baselines."""

    dataset = load_dataset(dataset_name)

    baselines = {
        # Traditional methods
        'KNN (k=5)': run_knn_baseline,
        'Isolation Forest': run_isolation_forest_baseline,
        'LSTM-VAE': run_lstm_vae_baseline,

        # Foundation model baselines
        'TimesFM Only': run_timesfm_only_baseline,
        'Chronos Only': run_chronos_only_baseline,
        'Statistical Evidence': run_statistical_evidence_baseline,

        # Phase 3 variants
        'Phase3 (No RAG)': run_phase3_no_rag,
        'Phase3 (No LLM)': run_phase3_statistical,
        'Phase3 (Full)': run_phase3_full,
    }

    results = []
    evaluator = Evaluator()

    for name, baseline_func in baselines.items():
        print(f"Running {name}...")
        try:
            predictions = baseline_func(dataset)
            metrics = evaluator.evaluate(dataset.y_test, predictions)

            results.append({
                'Method': name,
                'F1': metrics.f1,
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'PA-F1': metrics.pa_f1,
                'VUS-PR': metrics.vus_pr
            })
        except Exception as e:
            print(f"  Failed: {e}")

    return pd.DataFrame(results)
```

**1.4 Statistical Significance Testing**

**File**: `experiments/statistical_testing.py`
```python
from scipy.stats import wilcoxon, friedmanchisquare
import numpy as np

def compare_methods_significance(
    method1_scores: List[float],
    method2_scores: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Test if method1 is significantly better than method2.

    Args:
        method1_scores: F1 scores across datasets for method 1
        method2_scores: F1 scores across datasets for method 2
        alpha: Significance level

    Returns:
        {
            'statistic': float,
            'p_value': float,
            'significant': bool,
            'winner': str
        }
    """
    # Wilcoxon signed-rank test
    stat, p_value = wilcoxon(method1_scores, method2_scores)

    return {
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'winner': 'method1' if np.mean(method1_scores) > np.mean(method2_scores) else 'method2'
    }
```

**Week 1 Success Criteria**:
- [ ] All benchmark datasets prepared and validated
- [ ] Comprehensive evaluation completed on all datasets
- [ ] Baseline comparisons show Phase 3 performance
- [ ] Statistical significance tests completed
- [ ] Results documented in tables and visualizations

---

### Week 2: Ablation Studies & Optimization

**Deliverable**: Understanding of critical components and optimized system

#### Tasks

**2.1 Ablation Studies**

**File**: `experiments/ablation_studies.py`
```python
#!/usr/bin/env python
"""Ablation studies to understand component contributions."""

def ablation_foundation_models():
    """Test TimesFM vs Chronos vs Ensemble."""

    configs = [
        'configs/ablation/timesfm_only.yaml',
        'configs/ablation/chronos_only.yaml',
        'configs/ablation/ensemble.yaml'
    ]

    # Run and compare
    results = run_ablation(configs, datasets=['msl', 'smap', 'yahoo'])

    # Conclusion: Is ensemble worth the cost?
    return analyze_ablation_results(results)

def ablation_evidence_metrics():
    """Test importance of each evidence category."""

    evidence_combinations = [
        ['forecast_based'],  # Only forecast errors
        ['statistical_tests'],  # Only Z-score, Grubbs, etc.
        ['distribution_based'],  # Only KL, Wasserstein
        ['pattern_based'],  # Only ACF, volatility
        ['forecast_based', 'statistical_tests'],  # Best 2?
        ['all']  # Full evidence
    ]

    # Test each combination
    results = []
    for evidence_types in evidence_combinations:
        config = create_config_with_evidence(evidence_types)
        metrics = evaluate_config(config)
        results.append({
            'evidence_types': evidence_types,
            'f1': metrics.f1
        })

    # Conclusion: Which evidence matters most?
    return rank_evidence_importance(results)

def ablation_llm_models():
    """Compare LLM models: GPT-4 vs Gemini vs Claude."""

    llm_configs = {
        'GPT-4 Turbo': {'backend': 'openai', 'model': 'gpt-4-turbo'},
        'Gemini 1.5 Pro': {'backend': 'gemini', 'model': 'gemini-1.5-pro'},
        'Gemini 2.0 Flash': {'backend': 'gemini', 'model': 'gemini-2.0-flash'},
        'Claude 3 Opus': {'backend': 'anthropic', 'model': 'claude-3-opus'}
    }

    results = []
    for name, llm_config in llm_configs.items():
        metrics, cost = evaluate_with_llm(llm_config)
        results.append({
            'model': name,
            'f1': metrics.f1,
            'cost_per_1000_windows': cost
        })

    # Conclusion: Best quality/cost tradeoff?
    return analyze_llm_tradeoffs(results)

def ablation_rag_impact():
    """Test impact of RAG system."""

    configs = [
        'configs/ablation/no_rag.yaml',
        'configs/ablation/with_rag_k1.yaml',
        'configs/ablation/with_rag_k3.yaml',
        'configs/ablation/with_rag_k5.yaml',
    ]

    # Measure consistency and performance
    results = evaluate_consistency_and_performance(configs)

    # Conclusion: Does RAG improve reasoning?
    return analyze_rag_impact(results)

def run_all_ablations():
    """Run all ablation studies."""

    ablation_results = {
        'foundation_models': ablation_foundation_models(),
        'evidence_metrics': ablation_evidence_metrics(),
        'llm_models': ablation_llm_models(),
        'rag_impact': ablation_rag_impact()
    }

    # Generate report
    generate_ablation_report(ablation_results)

    return ablation_results
```

**2.2 Performance Optimization**

**File**: `experiments/performance_optimization.py`
```python
#!/usr/bin/env python
"""Optimize system performance."""

import time
import psutil
from memory_profiler import profile

@profile
def profile_pipeline_memory():
    """Profile memory usage of full pipeline."""

    pipeline = build_pipeline_from_config('configs/phase3_full.yaml')

    dataset = load_dataset('yahoo_s5')

    # Measure memory before
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    # Fit and predict
    pipeline.fit(dataset.X_train)
    result = pipeline.predict(dataset.X_test)

    # Measure memory after
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    print(f"Memory usage: {memory_after - memory_before:.2f} MB")

def optimize_batch_processing():
    """Test batch processing strategies."""

    # Test different batch sizes
    batch_sizes = [1, 10, 50, 100]

    results = []
    for batch_size in batch_sizes:
        start = time.time()

        # Process windows in batches
        process_in_batches(windows, batch_size=batch_size)

        elapsed = time.time() - start
        results.append({
            'batch_size': batch_size,
            'time': elapsed,
            'throughput': len(windows) / elapsed
        })

    return pd.DataFrame(results)

def optimize_llm_caching():
    """Test caching effectiveness."""

    # Run with cache disabled
    time_no_cache = benchmark_pipeline(use_cache=False)

    # Run with cache enabled
    time_with_cache = benchmark_pipeline(use_cache=True)

    speedup = time_no_cache / time_with_cache

    print(f"Speedup with caching: {speedup:.2f}x")
```

**2.3 Cost Analysis**

**File**: `experiments/cost_analysis.py`
```python
#!/usr/bin/env python
"""Analyze and optimize costs."""

def estimate_deployment_cost(
    num_windows_per_day: int,
    llm_model: str,
    use_cost_optimizer: bool = True
):
    """Estimate monthly deployment cost."""

    # Token counts
    tokens_per_window = 1500  # Prompt + completion

    # Pricing (per million tokens)
    pricing = {
        'gpt-4-turbo': 20.0,  # Average of input/output
        'gemini-1.5-pro': 5.25,
        'gemini-2.0-flash': 0.375,
        'claude-3-opus': 45.0
    }

    # Calculate base cost
    windows_per_month = num_windows_per_day * 30
    total_tokens = windows_per_month * tokens_per_window
    base_cost = (total_tokens / 1_000_000) * pricing[llm_model]

    # Apply cost optimizer savings
    if use_cost_optimizer:
        # Optimizer reduces LLM calls by ~60% (clear cases use statistical baseline)
        actual_cost = base_cost * 0.4
        savings = base_cost - actual_cost
    else:
        actual_cost = base_cost
        savings = 0

    return {
        'base_cost_usd': base_cost,
        'actual_cost_usd': actual_cost,
        'savings_usd': savings,
        'cost_per_window': actual_cost / windows_per_month
    }

# Example usage
print("Monthly cost for 10,000 windows/day:")
for model in ['gpt-4-turbo', 'gemini-1.5-pro', 'gemini-2.0-flash']:
    cost = estimate_deployment_cost(10000, model, use_cost_optimizer=True)
    print(f"{model}: ${cost['actual_cost_usd']:.2f}/month")
```

**2.4 Production Readiness Checklist**

**File**: `deployment/production_checklist.md`
```markdown
# Production Deployment Checklist

## Performance
- [ ] Pipeline processes 1000 windows in < 10 minutes
- [ ] Memory usage < 4GB for full pipeline
- [ ] Latency p99 < 5 seconds per window

## Reliability
- [ ] Error handling for all API failures
- [ ] Automatic retry with exponential backoff
- [ ] Fallback to statistical baseline if LLM fails
- [ ] Health check endpoint implemented
- [ ] Monitoring and alerting configured

## Cost Optimization
- [ ] Cost optimizer reduces API calls by 50%+
- [ ] Response caching implemented
- [ ] Batch processing for efficiency
- [ ] Cost tracking and budgets configured

## Security
- [ ] API keys stored in environment variables
- [ ] No sensitive data in logs
- [ ] Input validation and sanitization
- [ ] Rate limiting implemented

## Documentation
- [ ] API documentation complete
- [ ] Deployment guide written
- [ ] Troubleshooting guide created
- [ ] Example configurations provided

## Testing
- [ ] Unit tests pass (>80% coverage)
- [ ] Integration tests pass
- [ ] Load testing completed
- [ ] Validated on benchmark datasets
```

**Week 2 Success Criteria**:
- [ ] Ablation studies identify critical components
- [ ] Performance optimizations reduce runtime by 30%+
- [ ] Cost analysis shows deployment feasibility
- [ ] Production readiness checklist complete
- [ ] System ready for deployment

---

## Phase 4 Completion Checklist

### Evaluation
- [ ] Comprehensive evaluation on 5+ datasets
- [ ] Baseline comparisons completed
- [ ] Statistical significance testing done
- [ ] Results documented with tables and visualizations
- [ ] F1 > 0.75 achieved on majority of datasets

### Ablation Studies
- [ ] Foundation model comparison (TimesFM vs Chronos vs Ensemble)
- [ ] Evidence metric importance ranking
- [ ] LLM model comparison (GPT-4 vs Gemini vs Claude)
- [ ] RAG impact analysis
- [ ] Key findings documented

### Optimization
- [ ] Performance profiling completed
- [ ] Memory optimization (< 4GB usage)
- [ ] Speed optimization (30%+ improvement)
- [ ] Cost optimization (50%+ reduction in API calls)
- [ ] Batch processing implemented

### Production Readiness
- [ ] Production checklist completed
- [ ] Deployment guide written
- [ ] Monitoring and alerting configured
- [ ] Error handling comprehensive
- [ ] Load testing passed

### Documentation
- [ ] Research paper / technical report drafted
- [ ] Experiment results documented
- [ ] Ablation study findings summarized
- [ ] Deployment guide complete
- [ ] API reference updated

---

## Expected Outcomes

By end of Phase 4, you should have:

1. **Validated System**: F1 > 0.75 on standard benchmarks with statistical significance
2. **Understanding**: Know which components are critical and why
3. **Optimized Performance**: 30%+ faster, 50%+ cheaper, production-ready
4. **Publishable Results**: Complete evaluation with comparisons to baselines
5. **Deployment Ready**: Production checklist complete, monitoring configured

**Research Contributions**:
- Enhanced 4-step architecture for foundation model + LLM approach
- Multi-faceted statistical evidence framework (10+ metrics)
- RAG-enhanced LLM reasoning for anomaly detection
- Zero-shot, explainable anomaly detection system

---

## Research Paper Outline

**File**: `docs/research_paper_outline.md`
```markdown
# Research Paper: Zero-Shot Explainable Time Series Anomaly Detection with Foundation Models and LLM Reasoning

## 1. Introduction
- Problem: Need for explainable, zero-shot anomaly detection
- Limitations of training-based approaches
- Our approach: Foundation models + Statistical evidence + LLM reasoning

## 2. Related Work
- Foundation models for time series (TimesFM, Chronos)
- LLM for time series analysis
- Traditional anomaly detection methods
- Explainable AI

## 3. Methodology
### 3.1 Enhanced 4-Step Pipeline
### 3.2 Foundation Model Forecasting (Step 1)
### 3.3 Statistical Evidence Extraction (Step 2)
### 3.4 LLM Reasoning with RAG (Step 3)
### 3.5 Post-Processing (Step 4)

## 4. Experiments
### 4.1 Datasets and Metrics
### 4.2 Baseline Comparisons
### 4.3 Ablation Studies
### 4.4 Qualitative Analysis (Explanations)

## 5. Results
### 5.1 Quantitative Results (F1, PA-F1, VUS-PR)
### 5.2 Comparison to SOTA
### 5.3 Ablation Findings
### 5.4 Example Explanations

## 6. Discussion
### 6.1 Critical Components
### 6.2 Limitations
### 6.3 Cost-Performance Tradeoffs
### 6.4 Future Work

## 7. Conclusion
```

---

## Next Steps After Phase 4

**Option 1: Research Publication**
- Submit to conference (ICML, NeurIPS, ICLR, KDD)
- Write full paper using results from Phase 4
- Create supplementary materials

**Option 2: Production Deployment**
- Deploy to cloud (AWS, GCP, Azure)
- Create web API for anomaly detection service
- Build monitoring dashboard

**Option 3: Open Source Release**
- Clean up codebase
- Write comprehensive documentation
- Release on GitHub with examples
- Create tutorial notebooks

**Option 4: Continue Research**
- Fine-tune foundation models on anomaly data
- Explore other LLM reasoning strategies
- Add causal analysis
- Multi-modal anomaly detection

---

**Status**: Planned
**Last Updated**: 2026-02-17
**Prerequisites**: Phase 3 complete, benchmarks prepared
