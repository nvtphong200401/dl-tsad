# Statistical Evidence Framework

## Position in Pipeline

This framework implements **Step 2: Detection via Evidence Extraction** in the 4-step pipeline (Phase 3).

**Pipeline Context:**
- **Step 1** (Preprocessing): Generates forecasts and quantiles via foundation models
- **Step 2** (Detection): **← THIS FRAMEWORK** - Extracts 10+ statistical anomaly signals
- **Step 3** (Scoring): LLM reasoning aggregates evidence into explainable scores
- **Step 4** (Post-Processing): Parses outputs and generates final predictions

**Key Insight**: Statistical evidence extraction IS an anomaly detection method. It identifies anomalous patterns by computing multiple independent metrics, implementing the `DetectionMethod` interface used in the pipeline.

## Overview

This document specifies a **multi-faceted statistical evidence framework** for time series anomaly detection. The framework extracts 10+ independent anomaly signals that are used to:
1. Ground LLM reasoning with quantitative evidence (Step 3)
2. Prevent hallucination by providing objective measurements
3. Enable fallback to pure statistical detection (without LLM)
4. Support ablation studies to identify most effective signals

## Design Principles

1. **Independence**: Each metric measures a different aspect of anomalies
2. **Interpretability**: Each metric has clear statistical meaning
3. **Robustness**: Use multiple signals to handle diverse anomaly types
4. **Modularity**: Easy to add/remove metrics without breaking pipeline
5. **Efficiency**: Compute all metrics in a single pass when possible

## Evidence Categories

### Category 1: Forecast-Based Evidence

These metrics compare actual values to foundation model predictions.

#### 1.1 Mean Absolute Error (MAE)
**Formula**: `MAE = mean(|actual - forecast|)`

**Interpretation**: Average magnitude of forecast errors. High MAE indicates poor predictability.

**Threshold Strategy**:
- Compute historical MAE on training windows
- Flag anomaly if `MAE > P95(historical_MAE)`
- Adaptive threshold adjusts to time series scale

**Implementation**:
```python
def compute_mae_evidence(actual, forecast, historical_mae):
    mae = np.mean(np.abs(actual - forecast))
    threshold = np.percentile(historical_mae, 95)
    is_anomalous = mae > threshold
    percentile_rank = percentileofscore(historical_mae, mae)

    return {
        'mae': mae,
        'mae_threshold': threshold,
        'mae_anomalous': is_anomalous,
        'mae_percentile': percentile_rank
    }
```

**Anomaly Types Detected**: Point anomalies, pattern changes, level shifts

#### 1.2 Mean Squared Error (MSE)
**Formula**: `MSE = mean((actual - forecast)^2)`

**Interpretation**: Emphasizes large errors (squares penalize outliers more). Sensitive to extreme points.

**Threshold Strategy**: Same as MAE, use P95 of historical MSE

**Anomaly Types Detected**: Extreme point anomalies, volatility spikes

#### 1.3 Mean Absolute Percentage Error (MAPE)
**Formula**: `MAPE = mean(|actual - forecast| / |actual|) * 100`

**Interpretation**: Scale-independent error metric. Useful for comparing across time series.

**Threshold Strategy**: Use P95 of historical MAPE

**Caution**: Undefined when `actual = 0`, handle with epsilon or skip

**Anomaly Types Detected**: Relative deviations, useful for trend anomalies

#### 1.4 Quantile Violations
**Definition**: Actual value falls outside confidence interval from probabilistic forecast

**Computation** (requires Chronos):
```python
def compute_quantile_violations(actual, forecast_samples):
    # forecast_samples: [num_samples, forecast_length]
    quantiles = np.quantile(forecast_samples, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)

    violations = {
        'below_p01': actual < quantiles[0],  # Extremely low
        'below_p10': actual < quantiles[1],  # Low
        'above_p90': actual > quantiles[3],  # High
        'above_p99': actual > quantiles[4],  # Extremely high
    }

    # Confidence interval width (indicator of uncertainty)
    ci_width = quantiles[4] - quantiles[0]  # P99 - P01

    return {
        'quantile_violations': violations,
        'ci_width': ci_width,
        'any_extreme_violation': violations['below_p01'].any() or violations['above_p99'].any()
    }
```

**Anomaly Types Detected**: Extreme values, distribution tail events

#### 1.5 Surprise Score
**Formula**: `Surprise = -log P(actual | forecast_distribution)`

**Interpretation**: Negative log-likelihood of actual value under predictive distribution. High surprise = unlikely observation.

**Computation**:
```python
def compute_surprise(actual, forecast_samples):
    # Estimate distribution using KDE
    from scipy.stats import gaussian_kde

    surprise_scores = []
    for t in range(len(actual)):
        kde = gaussian_kde(forecast_samples[:, t])
        likelihood = kde.pdf(actual[t])
        surprise = -np.log(likelihood + 1e-10)  # Add epsilon for stability
        surprise_scores.append(surprise)

    return {
        'surprise': np.array(surprise_scores),
        'mean_surprise': np.mean(surprise_scores),
        'max_surprise': np.max(surprise_scores),
        'high_surprise_count': np.sum(surprise_scores > threshold)
    }
```

**Threshold Strategy**: Use P95 of historical surprise scores

**Anomaly Types Detected**: Low-probability events, distribution outliers

---

### Category 2: Statistical Test Evidence

Classical statistical tests for outliers and distributional changes.

#### 2.1 Z-Score
**Formula**: `z = (x - mean) / std`

**Interpretation**: Number of standard deviations from mean. Classic outlier detection.

**Threshold Strategy**: `|z| > 3` (3-sigma rule) or adaptive based on data distribution

**Implementation**:
```python
def compute_z_score_evidence(window, historical_mean, historical_std):
    z_scores = (window - historical_mean) / (historical_std + 1e-10)

    return {
        'z_scores': z_scores,
        'max_z_score': np.max(np.abs(z_scores)),
        'extreme_z_count': np.sum(np.abs(z_scores) > 3),
        'anomalous': np.abs(z_scores) > 3
    }
```

**Anomaly Types Detected**: Point anomalies, extreme values

#### 2.2 Grubbs Test
**Purpose**: Detect outliers in univariate data

**Formula**: `G = max(|x_i - mean|) / std`

**Critical Value**: Depends on sample size and significance level (α)

**Implementation**:
```python
from scipy.stats import t

def grubbs_test(data, alpha=0.05):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    # Grubbs statistic
    G = np.max(np.abs(data - mean)) / std

    # Critical value
    t_crit = t.ppf(1 - alpha / (2 * n), n - 2)
    G_crit = ((n - 1) * np.sqrt(t_crit**2)) / np.sqrt(n * (n - 2 + t_crit**2))

    return {
        'grubbs_statistic': G,
        'grubbs_critical': G_crit,
        'is_outlier': G > G_crit,
        'outlier_index': np.argmax(np.abs(data - mean))
    }
```

**Anomaly Types Detected**: Single outlier in window

#### 2.3 CUSUM (Cumulative Sum)
**Purpose**: Detect shifts in mean level (change point detection)

**Formula**:
```
S[i] = max(0, S[i-1] + (x[i] - target) - slack)
```

**Implementation**:
```python
def compute_cusum(data, target=None, slack=1.0):
    if target is None:
        target = np.mean(data)

    cusum_pos = np.zeros(len(data))
    cusum_neg = np.zeros(len(data))

    for i in range(1, len(data)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + (data[i] - target) - slack)
        cusum_neg[i] = max(0, cusum_neg[i-1] - (data[i] - target) - slack)

    # Detect change points
    threshold = 5 * np.std(data)
    change_points = (cusum_pos > threshold) | (cusum_neg > threshold)

    return {
        'cusum_pos': cusum_pos,
        'cusum_neg': cusum_neg,
        'change_points': change_points,
        'max_cusum': max(np.max(cusum_pos), np.max(cusum_neg))
    }
```

**Anomaly Types Detected**: Level shifts, mean changes, trend breaks

---

### Category 3: Distribution-Based Evidence

Compare distributions between training and test windows.

#### 3.1 KL Divergence
**Formula**: `KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))`

**Interpretation**: Measures how much test distribution Q diverges from training distribution P

**Implementation**:
```python
from scipy.stats import entropy

def compute_kl_divergence(train_data, test_data, bins=20):
    # Create histograms
    hist_train, bin_edges = np.histogram(train_data, bins=bins, density=True)
    hist_test, _ = np.histogram(test_data, bins=bin_edges, density=True)

    # Add small epsilon to avoid log(0)
    hist_train = hist_train + 1e-10
    hist_test = hist_test + 1e-10

    # Compute KL divergence
    kl_div = entropy(hist_train, hist_test)

    return {
        'kl_divergence': kl_div,
        'high_divergence': kl_div > threshold
    }
```

**Threshold Strategy**: Use P95 of historical KL divergences between consecutive train windows

**Anomaly Types Detected**: Distributional shifts, collective anomalies

#### 3.2 Wasserstein Distance
**Formula**: Earth Mover's Distance between distributions

**Interpretation**: Minimum cost to transform one distribution into another. More robust than KL divergence.

**Implementation**:
```python
from scipy.stats import wasserstein_distance

def compute_wasserstein_evidence(train_data, test_data):
    w_dist = wasserstein_distance(train_data, test_data)

    # Normalize by scale
    scale = np.std(train_data)
    normalized_w_dist = w_dist / (scale + 1e-10)

    return {
        'wasserstein_distance': w_dist,
        'normalized_wasserstein': normalized_w_dist,
        'high_distance': normalized_w_dist > threshold
    }
```

**Threshold Strategy**: Use P95 of historical normalized Wasserstein distances

**Anomaly Types Detected**: Distribution shifts, collective anomalies

---

### Category 4: Pattern-Based Evidence

Detect changes in time series patterns and structure.

#### 4.1 Autocorrelation Break
**Purpose**: Detect changes in periodicity or serial correlation

**Formula**: Compare ACF (autocorrelation function) between train and test

**Implementation**:
```python
from statsmodels.tsa.stattools import acf

def compute_acf_break(train_data, test_data, nlags=20):
    acf_train = acf(train_data, nlags=nlags, fft=True)
    acf_test = acf(test_data, nlags=nlags, fft=True)

    # Compute difference
    acf_diff = np.abs(acf_train - acf_test)
    max_diff = np.max(acf_diff)
    mean_diff = np.mean(acf_diff)

    # Detect significant period (peak in ACF)
    train_period = np.argmax(acf_train[1:]) + 1 if len(acf_train) > 1 else None
    test_period = np.argmax(acf_test[1:]) + 1 if len(acf_test) > 1 else None
    period_changed = train_period != test_period

    return {
        'acf_diff': acf_diff,
        'max_acf_diff': max_diff,
        'mean_acf_diff': mean_diff,
        'period_changed': period_changed,
        'train_period': train_period,
        'test_period': test_period
    }
```

**Threshold Strategy**: `max_acf_diff > 0.3` (empirical threshold)

**Anomaly Types Detected**: Periodicity changes, pattern disruptions

#### 4.2 Volatility Spike
**Purpose**: Detect sudden changes in variance

**Formula**: `Volatility Ratio = std(test) / std(train)`

**Implementation**:
```python
def compute_volatility_evidence(train_data, test_data):
    std_train = np.std(train_data)
    std_test = np.std(test_data)

    volatility_ratio = std_test / (std_train + 1e-10)

    # Also compute rolling volatility within test window
    from pandas import Series
    rolling_std = Series(test_data).rolling(window=10).std()
    max_rolling_std = rolling_std.max()

    return {
        'std_train': std_train,
        'std_test': std_test,
        'volatility_ratio': volatility_ratio,
        'max_rolling_std': max_rolling_std,
        'high_volatility': volatility_ratio > 2.0,  # 2x increase
        'extreme_volatility': volatility_ratio > 5.0  # 5x increase
    }
```

**Threshold Strategy**: `volatility_ratio > 2.0` for high, `> 5.0` for extreme

**Anomaly Types Detected**: Volatility spikes, variance changes

#### 4.3 Trend Break
**Purpose**: Detect sudden level shifts or trend changes

**Method**: Fit piecewise linear regression and detect breakpoints

**Implementation**:
```python
from scipy.stats import linregress

def compute_trend_break(data, window_size=20):
    n = len(data)
    if n < 2 * window_size:
        return {'trend_break': False, 'breakpoint': None}

    # Fit linear trend to first half and second half
    mid = n // 2
    slope1, intercept1, _, _, _ = linregress(range(mid), data[:mid])
    slope2, intercept2, _, _, _ = linregress(range(mid, n), data[mid:])

    # Compare slopes and levels
    slope_diff = abs(slope2 - slope1)
    level_diff = abs(intercept2 - intercept1)

    # Detect breakpoint
    trend_break = (slope_diff > threshold_slope) or (level_diff > threshold_level)

    return {
        'slope_before': slope1,
        'slope_after': slope2,
        'slope_diff': slope_diff,
        'level_diff': level_diff,
        'trend_break': trend_break,
        'breakpoint': mid if trend_break else None
    }
```

**Threshold Strategy**: Adaptive based on training data variability

**Anomaly Types Detected**: Level shifts, trend changes

---

### Category 5: Optional Pre-trained Model Evidence

Use pre-trained deep learning models from Phase 2 as additional signals (inference only, no training).

#### 5.1 AER BiLSTM Score
**Model**: Adversarial Encoder-Reconstructor with BiLSTM

**Usage**: Load pre-trained weights, run forward pass

**Implementation**:
```python
import torch
from src.models.aer import AER

def compute_aer_evidence(window, model_path):
    # Load pre-trained model
    model = AER(window_size=len(window), input_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Run inference (no training)
    with torch.no_grad():
        window_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(-1)
        aer_score = model(window_tensor).item()

    return {
        'aer_score': aer_score,
        'aer_anomalous': aer_score > threshold
    }
```

**Threshold Strategy**: Use P95 of AER scores on training data

**Anomaly Types Detected**: Complex patterns learned during training

#### 5.2 Anomaly Transformer Score
**Model**: Transformer with anomaly attention mechanism

**Usage**: Same as AER (load pre-trained, inference only)

**Implementation**: Similar to AER, see `src/models/anomaly_transformer.py`

**Note**: These are optional. The pipeline works without them, but they can provide additional signal if weights are available.

---

## Evidence Aggregation

### Evidence Dictionary Structure

Each window produces a comprehensive evidence dictionary:

```python
evidence = {
    # Forecast-based
    'mae': 2.34,
    'mae_percentile': 95.2,
    'mae_anomalous': True,
    'mse': 6.78,
    'quantile_violations': {'above_p99': True, 'below_p01': False},
    'surprise_score': 8.5,

    # Statistical tests
    'z_score': 3.8,
    'extreme_z_count': 2,
    'grubbs_statistic': 3.2,
    'grubbs_outlier': True,
    'cusum_max': 12.5,
    'cusum_change_points': [42, 48],

    # Distribution
    'kl_divergence': 0.45,
    'wasserstein_distance': 2.1,

    # Pattern
    'acf_max_diff': 0.35,
    'period_changed': True,
    'volatility_ratio': 5.2,
    'trend_break': True,

    # Optional
    'aer_score': 0.82,
    'transformer_score': 0.76
}
```

### Weighting Schemes

Different strategies for combining evidence:

#### 1. Majority Voting
Simple: Anomaly if > 50% of metrics exceed thresholds

```python
def majority_vote(evidence, thresholds):
    votes = []
    for metric, value in evidence.items():
        if metric in thresholds:
            votes.append(value > thresholds[metric])
    return sum(votes) > len(votes) / 2
```

#### 2. Weighted Combination
Assign weights based on empirical performance:

```python
weights = {
    'mae': 0.15,
    'z_score': 0.20,
    'quantile_violations': 0.15,
    'volatility_ratio': 0.10,
    'aer_score': 0.10,
    # ... other metrics
}

def weighted_score(evidence, weights):
    score = sum(weights[k] * evidence[k] for k in weights if k in evidence)
    return score / sum(weights.values())
```

#### 3. LLM Reasoning (Recommended)
Let the LLM decide how to weight evidence based on context. This is the core of Phase 3.

---

## Implementation Guide

### Module Structure

```
src/statistical_evidence/
├── __init__.py
├── evidence_extractor.py       # Main class
├── forecast_based.py            # Forecast error metrics
├── statistical_tests.py         # Z-score, Grubbs, CUSUM
├── distribution_based.py        # KL, Wasserstein
├── pattern_based.py             # ACF, volatility, trend
└── pretrained_models.py         # Optional AER/Transformer
```

### Main Extractor Class

```python
class StatisticalEvidenceExtractor:
    def __init__(self, config):
        self.config = config
        self.enabled_metrics = config.get('enabled_metrics', 'all')

    def extract(self, train_data, test_data, forecast, forecast_samples=None):
        evidence = {}

        # Category 1: Forecast-based
        if 'forecast_based' in self.enabled_metrics:
            evidence.update(self._forecast_evidence(test_data, forecast, forecast_samples))

        # Category 2: Statistical tests
        if 'statistical_tests' in self.enabled_metrics:
            evidence.update(self._statistical_tests(test_data, train_data))

        # Category 3: Distribution-based
        if 'distribution_based' in self.enabled_metrics:
            evidence.update(self._distribution_evidence(train_data, test_data))

        # Category 4: Pattern-based
        if 'pattern_based' in self.enabled_metrics:
            evidence.update(self._pattern_evidence(train_data, test_data))

        # Category 5: Optional pre-trained models
        if 'pretrained_models' in self.enabled_metrics and self.config.get('use_pretrained'):
            evidence.update(self._pretrained_evidence(test_data))

        return evidence
```

### Configuration Example

```yaml
statistical_evidence:
  enabled_metrics:
    - forecast_based
    - statistical_tests
    - distribution_based
    - pattern_based
    - pretrained_models  # Optional

  thresholds:
    mae_percentile: 95
    z_score: 3.0
    volatility_ratio: 2.0
    kl_divergence: 0.5

  pretrained_models:
    use_aer: true
    aer_weights: "pretrained/aer_weights.pth"
    use_transformer: false
```

## Evaluation and Ablation

### Ablation Study Design

Test contribution of each evidence category:

1. **Baseline**: Random guessing (F1 ≈ 0.5)
2. **Forecast-only**: MAE, MSE, MAPE, quantiles, surprise
3. **Stats-only**: Z-score, Grubbs, CUSUM
4. **Distribution-only**: KL, Wasserstein
5. **Pattern-only**: ACF, volatility, trend
6. **Pretrained-only**: AER, Transformer
7. **Full**: All categories combined

### Metric Importance Analysis

For each metric, compute:
- **Individual F1**: Use only this metric with threshold
- **Ablation F1**: Full system minus this metric
- **Contribution**: `Full F1 - Ablation F1`

This identifies the most valuable signals.

### Cross-Dataset Validation

Test generalization across different datasets:
- UCR Anomaly Archive
- Yahoo S5
- NAB (Numenta Anomaly Benchmark)
- Custom datasets

Metrics that work consistently across datasets are more valuable.

## Future Extensions

### Adaptive Thresholds
- Learn optimal thresholds from training data
- Use quantile-based dynamic thresholds
- Bayesian updating of thresholds

### Domain-Specific Evidence
- Add domain knowledge (e.g., sensor ranges for IoT)
- Custom metrics for specific applications
- Incorporate external context (e.g., calendar events)

### Deep Learning Evidence
- Use foundation model attention weights
- Analyze intermediate representations
- Uncertainty decomposition (aleatoric vs epistemic)

---

**Status**: Specification complete, ready for implementation
**Last Updated**: 2026-02-17
**Dependencies**: Foundation models (TimesFM, Chronos), scipy, statsmodels, numpy
