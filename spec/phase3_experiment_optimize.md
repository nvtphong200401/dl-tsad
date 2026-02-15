# Phase 3: Experiment & Optimize (Week 4)

## Objectives

Systematically experiment with different component combinations to find the best configuration for each dataset and optimize the overall system. This phase focuses on:

1. ✅ Ablation studies - which components matter most?
2. ✅ Hyperparameter optimization
3. ✅ Component combination experiments
4. ✅ Performance analysis and visualization
5. ✅ LLM agent integration (optional)
6. ✅ Production deployment considerations

**Goal**: By end of Phase 3, achieve F1 > 0.80 through optimized configurations and identify best practices for time series anomaly detection.

---

## Deliverables

### 1. Ablation Studies

#### 1.1 Step-by-Step Ablation

**Purpose**: Understand which pipeline step contributes most to performance.

**File**: `experiments/ablation_studies.py`

```python
#!/usr/bin/env python
"""Ablation studies to understand component importance"""

import itertools
from typing import Dict, List
import pandas as pd
from src.utils.config_factory import build_pipeline_from_config
from src.data.loader import load_msl_smap_dataset
from src.evaluation.evaluator import Evaluator

def ablation_data_processing():
    """Test different data processing methods"""

    processors = [
        "RawWindowProcessor",
        "StatisticalFeatureProcessor",
        "NeuralEmbeddingProcessor",
        "AERProcessor",
        "AnomalyTransformerProcessor"
    ]

    # Keep detection, scoring, postprocessing constant
    base_config = {
        "detection": {"type": "DistanceBasedDetection", "params": {"k": 5}},
        "scoring": {"type": "MaxPoolingScoring"},
        "postprocessing": {
            "threshold": {"type": "F1OptimalThreshold"},
            "min_anomaly_length": 3,
            "merge_gap": 5
        }
    }

    results = []
    dataset = load_msl_smap_dataset("MSL")

    for processor in processors:
        print(f"\nTesting processor: {processor}")

        config = {
            "experiment": {"name": f"ablation_processor_{processor}"},
            "data_processing": {"type": processor, "window_size": 100, "stride": 1},
            **base_config
        }

        pipeline = build_pipeline_from_config(config)
        pipeline.fit(dataset.X_train)
        result = pipeline.predict(dataset.X_test, dataset.y_test)

        evaluator = Evaluator()
        eval_result = evaluator.evaluate(dataset.y_test, result.predictions, result.point_scores)

        results.append({
            "Component": "Data Processing",
            "Type": processor,
            "F1": eval_result.f1,
            "PA-F1": eval_result.pa_f1,
            "VUS-PR": eval_result.vus_pr
        })

    return pd.DataFrame(results)


def ablation_detection_method():
    """Test different detection methods"""

    detection_methods = [
        {"type": "DistanceBasedDetection", "params": {"k": 5}},
        {"type": "ReconstructionBasedDetection"},
        {"type": "PredictionBasedDetection"},
        {"type": "HybridDetection", "params": {"alpha": 0.5}},
        {"type": "ClassificationBasedDetection"}
    ]

    # Keep other components constant (use best from previous ablation)
    results = []
    # ... similar structure

    return pd.DataFrame(results)


def ablation_scoring_method():
    """Test different scoring methods"""

    scoring_methods = [
        "MaxPoolingScoring",
        "AveragePoolingScoring",
        "WeightedAverageScoring",
        "GaussianSmoothingScoring"
    ]

    results = []
    # ... similar structure

    return pd.DataFrame(results)


def ablation_postprocessing():
    """Test different post-processing configurations"""

    threshold_methods = [
        {"type": "PercentileThreshold", "params": {"percentile": 95}},
        {"type": "PercentileThreshold", "params": {"percentile": 99}},
        {"type": "F1OptimalThreshold"},
        {"type": "StatisticalThreshold", "params": {"k": 3}}
    ]

    min_lengths = [1, 3, 5, 10]
    merge_gaps = [0, 3, 5, 10]

    results = []
    # ... test combinations

    return pd.DataFrame(results)


def run_all_ablations():
    """Run all ablation studies and generate report"""

    print("=" * 60)
    print("ABLATION STUDY 1: Data Processing")
    print("=" * 60)
    df_processing = ablation_data_processing()
    print(df_processing.to_string(index=False))

    print("\n" + "=" * 60)
    print("ABLATION STUDY 2: Detection Method")
    print("=" * 60)
    df_detection = ablation_detection_method()
    print(df_detection.to_string(index=False))

    print("\n" + "=" * 60)
    print("ABLATION STUDY 3: Scoring Method")
    print("=" * 60)
    df_scoring = ablation_scoring_method()
    print(df_scoring.to_string(index=False))

    print("\n" + "=" * 60)
    print("ABLATION STUDY 4: Post-processing")
    print("=" * 60)
    df_postproc = ablation_postprocessing()
    print(df_postproc.to_string(index=False))

    # Combine and save
    df_all = pd.concat([df_processing, df_detection, df_scoring, df_postproc])
    df_all.to_csv("experiments/results_ablation_studies.csv", index=False)

    print("\nAblation studies complete! Results saved.")


if __name__ == "__main__":
    run_all_ablations()
```

---

### 2. Hyperparameter Optimization

#### 2.1 Grid Search for Key Hyperparameters

**File**: `experiments/hyperparameter_optimization.py`

```python
#!/usr/bin/env python
"""Hyperparameter optimization using grid search"""

import numpy as np
from itertools import product
import pandas as pd
from src.utils.config_factory import build_pipeline_from_config
from src.data.loader import load_msl_smap_dataset
from src.evaluation.evaluator import Evaluator

def optimize_aer_hyperparameters():
    """Grid search for AER hyperparameters"""

    # Hyperparameter grid
    param_grid = {
        "window_size": [50, 100, 150],
        "hidden_dim": [64, 128, 256],
        "num_layers": [1, 2, 3],
        "alpha": [0.3, 0.5, 0.7]  # Reconstruction vs prediction weight
    }

    dataset = load_msl_smap_dataset("MSL")
    results = []

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combination in product(*values):
        params = dict(zip(keys, combination))

        print(f"\nTesting params: {params}")

        # Build config
        config = {
            "experiment": {"name": "aer_hyperparam_search"},
            "data_processing": {
                "type": "AERProcessor",
                "window_size": params["window_size"],
                "stride": 1,
                "params": {
                    "hidden_dim": params["hidden_dim"],
                    "num_layers": params["num_layers"],
                    "alpha": params["alpha"]
                }
            },
            "detection": {
                "type": "HybridDetection",
                "params": {"alpha": params["alpha"]}
            },
            "scoring": {"type": "WeightedAverageScoring"},
            "postprocessing": {
                "threshold": {"type": "F1OptimalThreshold"},
                "min_anomaly_length": 3,
                "merge_gap": 5
            }
        }

        try:
            pipeline = build_pipeline_from_config(config)
            pipeline.fit(dataset.X_train)
            result = pipeline.predict(dataset.X_test, dataset.y_test)

            evaluator = Evaluator()
            eval_result = evaluator.evaluate(dataset.y_test, result.predictions, result.point_scores)

            results.append({
                **params,
                "F1": eval_result.f1,
                "PA-F1": eval_result.pa_f1,
                "VUS-PR": eval_result.vus_pr,
                "Training_Time": pipeline.execution_time.get("step1_fit", 0)
            })

            print(f"  F1: {eval_result.f1:.3f}")

        except Exception as e:
            print(f"  Failed: {e}")
            continue

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("F1", ascending=False)

    print("\n" + "=" * 60)
    print("TOP 5 CONFIGURATIONS:")
    print("=" * 60)
    print(df_results.head(5).to_string(index=False))

    df_results.to_csv("experiments/results_aer_hyperparameter_search.csv", index=False)

    return df_results


def optimize_transformer_hyperparameters():
    """Grid search for Anomaly Transformer hyperparameters"""

    param_grid = {
        "window_size": [50, 100, 150],
        "d_model": [256, 512],
        "n_heads": [4, 8],
        "n_layers": [2, 3, 4]
    }

    # Similar implementation
    pass


def optimize_per_dataset():
    """Find best hyperparameters for each dataset"""

    datasets = {
        "MSL": load_msl_smap_dataset("MSL"),
        "SMAP": load_msl_smap_dataset("SMAP")
    }

    best_configs = {}

    for dataset_name, dataset in datasets.items():
        print(f"\n{'='*60}")
        print(f"Optimizing for dataset: {dataset_name}")
        print(f"{'='*60}")

        results = optimize_aer_hyperparameters()  # Or transformer
        best = results.iloc[0]

        best_configs[dataset_name] = best.to_dict()

        print(f"\nBest config for {dataset_name}:")
        print(best)

    # Save best configs
    import yaml
    with open("configs/optimized/best_configs_per_dataset.yaml", "w") as f:
        yaml.dump(best_configs, f)

    return best_configs


if __name__ == "__main__":
    # Run optimization
    optimize_aer_hyperparameters()
    optimize_transformer_hyperparameters()
    optimize_per_dataset()
```

#### 2.2 Bayesian Optimization (Optional)

For more efficient search, integrate Optuna:

```python
import optuna

def objective_aer(trial):
    """Optuna objective function for AER"""

    # Suggest hyperparameters
    window_size = trial.suggest_int("window_size", 50, 200, step=25)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=64)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    alpha = trial.suggest_float("alpha", 0.2, 0.8)

    # Build and evaluate pipeline
    # ... (similar to grid search)

    return eval_result.f1  # Maximize F1


def bayesian_optimization():
    """Run Bayesian optimization with Optuna"""

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_aer, n_trials=50)

    print("\nBest hyperparameters:")
    print(study.best_params)
    print(f"Best F1: {study.best_value:.3f}")

    # Visualize optimization history
    import optuna.visualization as vis
    fig = vis.plot_optimization_history(study)
    fig.write_html("experiments/optimization_history.html")
```

---

### 3. Component Combination Experiments

#### 3.1 Exhaustive Combination Testing

**File**: `experiments/combination_experiments.py`

```python
#!/usr/bin/env python
"""Test all component combinations to find best pipeline"""

from itertools import product
import pandas as pd

def test_all_combinations():
    """Test all valid component combinations"""

    # Define component options
    processors = ["AERProcessor", "AnomalyTransformerProcessor"]
    detections = ["HybridDetection", "AssociationDiscrepancyDetection"]
    scorings = ["MaxPoolingScoring", "WeightedAverageScoring", "GaussianSmoothingScoring"]
    thresholds = ["F1OptimalThreshold", "PercentileThreshold"]

    dataset = load_msl_smap_dataset("MSL")
    results = []

    # Test all combinations
    for proc, det, scor, thresh in product(processors, detections, scorings, thresholds):

        # Skip invalid combinations
        if proc == "AERProcessor" and det != "HybridDetection":
            continue
        if proc == "AnomalyTransformerProcessor" and det != "AssociationDiscrepancyDetection":
            continue

        print(f"\nTesting: {proc} + {det} + {scor} + {thresh}")

        config = {
            "experiment": {"name": f"combo_{proc}_{det}_{scor}_{thresh}"},
            "data_processing": {"type": proc, "window_size": 100},
            "detection": {"type": det},
            "scoring": {"type": scor},
            "postprocessing": {"threshold": {"type": thresh}, "min_anomaly_length": 3}
        }

        try:
            pipeline = build_pipeline_from_config(config)
            pipeline.fit(dataset.X_train)
            result = pipeline.predict(dataset.X_test, dataset.y_test)

            evaluator = Evaluator()
            eval_result = evaluator.evaluate(dataset.y_test, result.predictions, result.point_scores)

            results.append({
                "Processor": proc,
                "Detection": det,
                "Scoring": scor,
                "Threshold": thresh,
                "F1": eval_result.f1,
                "PA-F1": eval_result.pa_f1,
                "VUS-PR": eval_result.vus_pr,
                "Latency_ms": eval_result.latency_ms
            })

        except Exception as e:
            print(f"  Failed: {e}")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("F1", ascending=False)

    print("\n" + "=" * 60)
    print("TOP 10 COMBINATIONS:")
    print("=" * 60)
    print(df_results.head(10).to_string(index=False))

    df_results.to_csv("experiments/results_combination_experiments.csv", index=False)

    return df_results


if __name__ == "__main__":
    test_all_combinations()
```

---

### 4. Performance Analysis & Visualization

#### 4.1 Visualization Notebook

**File**: `notebooks/results_analysis.ipynb`

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df_ablation = pd.read_csv("../experiments/results_ablation_studies.csv")
df_hyperparam = pd.read_csv("../experiments/results_aer_hyperparameter_search.csv")
df_combinations = pd.read_csv("../experiments/results_combination_experiments.csv")

# 1. Ablation study visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Data processing comparison
ax = axes[0, 0]
subset = df_ablation[df_ablation["Component"] == "Data Processing"]
subset.plot(x="Type", y="F1", kind="bar", ax=ax, legend=False)
ax.set_title("Impact of Data Processing Method")
ax.set_ylabel("F1 Score")

# Plot 2: Detection method comparison
ax = axes[0, 1]
subset = df_ablation[df_ablation["Component"] == "Detection"]
subset.plot(x="Type", y="F1", kind="bar", ax=ax, legend=False)
ax.set_title("Impact of Detection Method")

# Plot 3: Scoring method comparison
ax = axes[1, 0]
subset = df_ablation[df_ablation["Component"] == "Scoring"]
subset.plot(x="Type", y="F1", kind="bar", ax=ax, legend=False)
ax.set_title("Impact of Scoring Method")

# Plot 4: Threshold method comparison
ax = axes[1, 1]
subset = df_ablation[df_ablation["Component"] == "Post-processing"]
subset.plot(x="Type", y="F1", kind="bar", ax=ax, legend=False)
ax.set_title("Impact of Threshold Method")

plt.tight_layout()
plt.savefig("../experiments/ablation_analysis.png", dpi=300)

# 2. Hyperparameter impact
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Window size impact
ax = axes[0, 0]
sns.boxplot(data=df_hyperparam, x="window_size", y="F1", ax=ax)
ax.set_title("Impact of Window Size")

# Hidden dim impact
ax = axes[0, 1]
sns.boxplot(data=df_hyperparam, x="hidden_dim", y="F1", ax=ax)
ax.set_title("Impact of Hidden Dimension")

# Num layers impact
ax = axes[1, 0]
sns.boxplot(data=df_hyperparam, x="num_layers", y="F1", ax=ax)
ax.set_title("Impact of Number of Layers")

# Alpha impact
ax = axes[1, 1]
sns.scatterplot(data=df_hyperparam, x="alpha", y="F1", ax=ax)
ax.set_title("Impact of Alpha (Recon vs Pred Weight)")

plt.tight_layout()
plt.savefig("../experiments/hyperparameter_analysis.png", dpi=300)

# 3. Component combinations heatmap
pivot = df_combinations.pivot_table(
    index="Processor",
    columns="Scoring",
    values="F1",
    aggfunc="mean"
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0.75)
plt.title("F1 Score: Processor vs Scoring Method")
plt.tight_layout()
plt.savefig("../experiments/combination_heatmap.png", dpi=300)
```

#### 4.2 Performance Report Generator

**File**: `experiments/generate_report.py`

```python
#!/usr/bin/env python
"""Generate comprehensive performance report"""

import pandas as pd
from jinja2 import Template

def generate_html_report():
    """Generate HTML report with all results"""

    # Load all results
    df_sota = pd.read_csv("experiments/results_sota_comparison.csv")
    df_ablation = pd.read_csv("experiments/results_ablation_studies.csv")
    df_combinations = pd.read_csv("experiments/results_combination_experiments.csv")

    # Best overall configuration
    best = df_combinations.iloc[0]

    # Template
    template = Template("""
    <html>
    <head>
        <title>Best TSAD - Performance Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            .highlight { background-color: #ffeb3b; }
        </style>
    </head>
    <body>
        <h1>Best TSAD - Performance Report</h1>

        <h2>Executive Summary</h2>
        <p><strong>Best Configuration Achieved:</strong></p>
        <ul>
            <li>Processor: {{ best.Processor }}</li>
            <li>Detection: {{ best.Detection }}</li>
            <li>Scoring: {{ best.Scoring }}</li>
            <li>Threshold: {{ best.Threshold }}</li>
            <li><strong>F1 Score: {{ "%.3f"|format(best.F1) }}</strong></li>
            <li><strong>PA-F1 Score: {{ "%.3f"|format(best['PA-F1']) }}</strong></li>
        </ul>

        <h2>SOTA Method Comparison</h2>
        {{ df_sota.to_html(index=False) }}

        <h2>Ablation Studies</h2>
        {{ df_ablation.to_html(index=False) }}

        <h2>Top 10 Configurations</h2>
        {{ df_combinations.head(10).to_html(index=False) }}

        <h2>Visualizations</h2>
        <img src="ablation_analysis.png" width="100%">
        <img src="hyperparameter_analysis.png" width="100%">
        <img src="combination_heatmap.png" width="100%">

    </body>
    </html>
    """)

    # Render
    html = template.render(
        best=best,
        df_sota=df_sota,
        df_ablation=df_ablation,
        df_combinations=df_combinations
    )

    with open("experiments/performance_report.html", "w") as f:
        f.write(html)

    print("Report generated: experiments/performance_report.html")


if __name__ == "__main__":
    generate_html_report()
```

---

### 5. LLM Agent Integration (Optional)

#### 5.1 LLM Explainer for Anomalies

**File**: `src/llm/anomaly_explainer.py`

```python
import anthropic
import numpy as np
from typing import Dict, Any

class AnomalyExplainer:
    """LLM-based explanation layer for detected anomalies"""

    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5-20250929"

    def explain_anomaly(self,
                       anomaly_window: np.ndarray,
                       anomaly_scores: np.ndarray,
                       metadata: Dict[str, Any]) -> str:
        """Generate natural language explanation for detected anomaly"""

        # Prepare context
        context = self._prepare_context(anomaly_window, anomaly_scores, metadata)

        # Call LLM
        prompt = f"""You are an expert in time series anomaly detection.

Detected Anomaly Information:
{context}

Please provide:
1. A brief description of the anomaly (2-3 sentences)
2. Potential root causes (top 3 most likely)
3. Severity assessment (1-5 scale, with reasoning)
4. Recommended actions

Be concise and actionable."""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        return message.content[0].text

    def _prepare_context(self, window, scores, metadata):
        """Format anomaly data for LLM"""
        return f"""
Anomaly Window Shape: {window.shape}
Anomaly Score Range: [{scores.min():.3f}, {scores.max():.3f}]
Detected Features: {metadata.get('feature_names', 'Unknown')}
Dataset: {metadata.get('dataset_name', 'Unknown')}

Statistical Summary:
- Mean: {window.mean(axis=0)}
- Std: {window.std(axis=0)}
- Max deviation from normal: {self._compute_deviation(window)}
"""

    def _compute_deviation(self, window):
        # Compute how much this deviates from expected normal behavior
        return "Significant deviation in dimensions [...]"
```

#### 5.2 Integration with Pipeline

**File**: `experiments/run_with_llm_explanation.py`

```python
#!/usr/bin/env python
"""Run pipeline with LLM explanations"""

from src.llm.anomaly_explainer import AnomalyExplainer
from src.pipeline.orchestrator import AnomalyDetectionPipeline
from src.utils.config_factory import build_pipeline_from_config
from src.data.loader import load_msl_smap_dataset

def run_with_explanations():
    """Run detection and generate explanations for top anomalies"""

    # Load config and data
    with open("configs/pipelines/aer_pipeline.yaml") as f:
        config = yaml.safe_load(f)

    dataset = load_msl_smap_dataset("MSL")

    # Build and run pipeline
    pipeline = build_pipeline_from_config(config)
    pipeline.fit(dataset.X_train)
    result = pipeline.predict(dataset.X_test, dataset.y_test)

    # Initialize explainer
    explainer = AnomalyExplainer()

    # Find top-K anomalies
    top_k_indices = np.argsort(result.point_scores)[-10:]  # Top 10

    print("\n" + "="*60)
    print("TOP ANOMALIES WITH EXPLANATIONS")
    print("="*60)

    for rank, idx in enumerate(top_k_indices[::-1], 1):
        print(f"\n--- Anomaly #{rank} (Index: {idx}) ---")
        print(f"Anomaly Score: {result.point_scores[idx]:.3f}")
        print(f"Ground Truth: {'ANOMALY' if dataset.y_test[idx] == 1 else 'NORMAL'}")

        # Extract window around anomaly
        window_start = max(0, idx - 50)
        window_end = min(len(dataset.X_test), idx + 50)
        anomaly_window = dataset.X_test[window_start:window_end]

        # Get explanation
        explanation = explainer.explain_anomaly(
            anomaly_window=anomaly_window,
            anomaly_scores=result.point_scores[window_start:window_end],
            metadata={
                "dataset_name": "MSL",
                "feature_names": ["Channel " + str(i) for i in range(dataset.X_test.shape[1])]
            }
        )

        print(f"\nExplanation:\n{explanation}")
        print("-" * 60)


if __name__ == "__main__":
    run_with_explanations()
```

---

### 6. Production Deployment Considerations

#### 6.1 Model Optimization

**File**: `src/deployment/model_optimizer.py`

```python
#!/usr/bin/env python
"""Optimize models for production deployment"""

import torch
import onnx
import onnxruntime as ort

def export_to_onnx(model: torch.nn.Module, input_shape: tuple, output_path: str):
    """Export PyTorch model to ONNX for faster inference"""

    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"Model exported to {output_path}")


def quantize_model(model_path: str, output_path: str):
    """Quantize model to INT8 for faster inference"""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_path,
        output_path,
        weight_type=QuantType.QInt8
    )

    print(f"Quantized model saved to {output_path}")


def benchmark_inference_speed(model_path: str, num_runs: int = 100):
    """Benchmark inference speed"""
    import time

    session = ort.InferenceSession(model_path)
    input_shape = session.get_inputs()[0].shape
    dummy_input = np.random.randn(1, *input_shape[1:]).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {'input': dummy_input})

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        session.run(None, {'input': dummy_input})
    elapsed = time.time() - start

    latency = (elapsed / num_runs) * 1000  # ms
    print(f"Average latency: {latency:.2f} ms")
    print(f"Throughput: {1000/latency:.1f} inferences/sec")
```

#### 6.2 REST API Server

**File**: `src/deployment/api_server.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from src.pipeline.orchestrator import AnomalyDetectionPipeline
from src.utils.config_factory import build_pipeline_from_config

app = FastAPI(title="Best TSAD API")

# Load pre-trained pipeline
pipeline = None

class TimeSeriesData(BaseModel):
    data: list[list[float]]  # Shape: (T, D)
    window_size: int = 100

class AnomalyResponse(BaseModel):
    predictions: list[int]
    scores: list[float]
    num_anomalies: int
    anomaly_indices: list[int]

@app.on_event("startup")
def load_model():
    global pipeline
    with open("configs/production/best_pipeline.yaml") as f:
        config = yaml.safe_load(f)
    pipeline = build_pipeline_from_config(config)
    # Load pre-trained weights
    pipeline.load("models/production/best_model.pkl")

@app.post("/detect", response_model=AnomalyResponse)
def detect_anomalies(data: TimeSeriesData):
    """Detect anomalies in time series data"""

    try:
        X = np.array(data.data)
        result = pipeline.predict(X)

        anomaly_indices = np.where(result.predictions == 1)[0].tolist()

        return AnomalyResponse(
            predictions=result.predictions.tolist(),
            scores=result.point_scores.tolist(),
            num_anomalies=len(anomaly_indices),
            anomaly_indices=anomaly_indices
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}
```

Run with: `uvicorn src.deployment.api_server:app --reload`

---

## Testing Strategy

### Performance Testing

**File**: `tests/test_performance.py`
- Benchmark inference latency
- Test memory usage
- Test with large datasets
- Verify throughput requirements

### Stress Testing

**File**: `tests/test_stress.py`
- Test with extreme anomaly ratios
- Test with very long sequences
- Test with high-dimensional data
- Test concurrent requests (for API)

---

## Development Timeline

### Week 4 (Days 22-28): Optimization & Deployment

- [ ] **Day 22**: Run ablation studies
- [ ] **Day 23**: Hyperparameter optimization
- [ ] **Day 24**: Component combination experiments
- [ ] **Day 25**: Performance analysis & visualization
- [ ] **Day 26**: LLM integration (optional)
- [ ] **Day 27**: Production optimization & API
- [ ] **Day 28**: Final report & documentation

---

## Success Criteria for Phase 3

1. ✅ **Achieve F1 > 0.80** through optimized configurations
2. ✅ **Identify best pipeline per dataset** - documented configurations
3. ✅ **Comprehensive analysis** - understand why certain combinations work
4. ✅ **Production-ready system** - API deployed, models optimized
5. ✅ **Complete documentation** - usage guides, best practices

---

## Deliverables Checklist

### Code
- [ ] `experiments/ablation_studies.py`
- [ ] `experiments/hyperparameter_optimization.py`
- [ ] `experiments/combination_experiments.py`
- [ ] `experiments/generate_report.py`
- [ ] `src/llm/anomaly_explainer.py` (optional)
- [ ] `src/deployment/model_optimizer.py`
- [ ] `src/deployment/api_server.py`

### Analysis
- [ ] Ablation study results (CSV)
- [ ] Hyperparameter search results (CSV)
- [ ] Combination experiment results (CSV)
- [ ] Visualization plots (PNG)
- [ ] Performance report (HTML)

### Configs
- [ ] `configs/optimized/best_configs_per_dataset.yaml`
- [ ] `configs/production/best_pipeline.yaml`

### Documentation
- [ ] Final project report
- [ ] API documentation
- [ ] Deployment guide
- [ ] Best practices guide

---

## Final Report Outline

**File**: `docs/final_report.md`

```markdown
# Best TSAD - Final Report

## Executive Summary
- Project goals achieved
- Best F1 scores attained
- Key findings

## Methodology
- 4-step pipeline architecture
- Components implemented
- Evaluation framework

## Results

### SOTA Method Comparison
- Table comparing AER, Anomaly Transformer, baseline
- Performance on each dataset

### Ablation Studies
- Impact of each pipeline step
- Which components matter most

### Optimization Results
- Best hyperparameters found
- Best component combinations
- Dataset-specific recommendations

## Insights & Learnings
- Why certain configurations work better
- Trade-offs (accuracy vs latency)
- Recommendations for practitioners

## Future Work
- Additional methods to try
- Real-time deployment considerations
- Integration with monitoring systems

## Conclusion
```

---

## Next Steps After Phase 3

1. **Research Paper**: Write up findings for publication
2. **Open Source**: Release on GitHub with documentation
3. **Real-World Deployment**: Deploy to production monitoring system
4. **Continuous Improvement**: Set up feedback loop for model retraining

---

## Notes

- **Parallelization**: Run experiments in parallel to save time
- **Experiment tracking**: Use wandb for better tracking (optional)
- **Reproducibility**: Set random seeds, document environment
- **Cost**: LLM integration is optional - focus on detection first
