# Pipeline Design Reference

Quick reference guide for designing and configuring time series anomaly detection pipelines.

## Configuration Format

All pipelines are configured via YAML files with this structure:

```yaml
experiment:
  name: "pipeline_name"
  description: "Optional description"

data_processing:
  type: "ProcessorType"
  window_size: 100
  stride: 1
  params:
    # Processor-specific parameters

detection:
  type: "DetectionType"
  params:
    # Detection-specific parameters

scoring:
  type: "ScoringType"
  params:
    # Scoring-specific parameters (optional)

postprocessing:
  threshold:
    type: "ThresholdType"
    params:
      # Threshold-specific parameters
  min_anomaly_length: 3
  merge_gap: 5
```

---

## Step 1: Data Processing

### RawWindowProcessor
**Best for**: Baseline, simple datasets
```yaml
data_processing:
  type: "RawWindowProcessor"
  window_size: 100
  stride: 1
```

**Output**: Normalized sliding windows
**Computational cost**: Low
**Memory**: Low

---

### StatisticalFeatureProcessor
**Best for**: Interpretability, small datasets
```yaml
data_processing:
  type: "StatisticalFeatureProcessor"
  window_size: 100
  stride: 1
```

**Output**: Statistical features (mean, std, skew, kurtosis, min, max)
**Computational cost**: Low
**Memory**: Low
**Features per window**: 6 × num_dimensions

---

### NeuralEmbeddingProcessor
**Best for**: Complex patterns, large datasets
```yaml
data_processing:
  type: "NeuralEmbeddingProcessor"
  window_size: 100
  stride: 1
  params:
    embedding_dim: 64
    hidden_dim: 128
    epochs: 50
```

**Output**: Neural embeddings
**Computational cost**: High (requires training)
**Memory**: Medium
**Requires**: GPU recommended

---

### AERProcessor (SOTA)
**Best for**: Multivariate time series, SOTA performance
```yaml
data_processing:
  type: "AERProcessor"
  window_size: 100
  stride: 1
  params:
    hidden_dim: 128        # BiLSTM hidden dimension
    num_layers: 2          # Number of LSTM layers
    dropout: 0.1           # Dropout rate
    alpha: 0.5             # Weight: reconstruction vs prediction (0-1)
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
```

**Output**: [original, reconstruction, forward_pred, backward_pred] concatenated
**Computational cost**: Very High (BiLSTM training)
**Memory**: High
**Requires**: GPU strongly recommended
**Training time**: ~30 min per dataset on GPU

**Hyperparameter guidance**:
- `hidden_dim`: 64 (small), 128 (medium), 256 (large)
- `num_layers`: 1-3 (more layers = more capacity but slower)
- `alpha`: 0.5 (balanced), 0.3 (favor prediction), 0.7 (favor reconstruction)

---

### AnomalyTransformerProcessor (SOTA)
**Best for**: Capturing long-range dependencies, attention interpretability
```yaml
data_processing:
  type: "AnomalyTransformerProcessor"
  window_size: 100
  stride: 1
  params:
    d_model: 512           # Transformer hidden dimension
    n_heads: 8             # Number of attention heads
    n_layers: 3            # Number of transformer layers
    dropout: 0.1
    epochs: 10             # Fast training!
    batch_size: 32
    learning_rate: 0.0001
```

**Output**: Association discrepancy matrices
**Computational cost**: High (transformer training)
**Memory**: Very High (attention requires O(n²) memory)
**Requires**: GPU required
**Training time**: ~10 min per dataset on GPU (much faster than AER!)

**Hyperparameter guidance**:
- `d_model`: 256 (small), 512 (medium), 768 (large)
- `n_heads`: Must divide d_model evenly (4, 8, or 16)
- `n_layers`: 2-4 (diminishing returns after 3)
- `window_size`: Larger windows capture longer dependencies but increase memory

---

## Step 2: Detection Method

### DistanceBasedDetection
**Best for**: Baseline, interpretability
```yaml
detection:
  type: "DistanceBasedDetection"
  params:
    method: "knn"          # "knn" or "lof"
    k: 5                   # Number of neighbors
```

**When to use**: Simple baseline, need interpretability
**Computational cost**: Medium (distance computation)
**Works with**: RawWindowProcessor, StatisticalFeatureProcessor

---

### ReconstructionBasedDetection
**Best for**: AutoEncoder-based models
```yaml
detection:
  type: "ReconstructionBasedDetection"
  params:
    metric: "mse"          # "mse" or "mae"
```

**When to use**: With NeuralEmbeddingProcessor
**Computational cost**: Low (just compute error)

---

### PredictionBasedDetection
**Best for**: LSTM/RNN-based models
```yaml
detection:
  type: "PredictionBasedDetection"
  params:
    metric: "mse"
```

**When to use**: Time series forecasting models
**Computational cost**: Low

---

### HybridDetection (SOTA)
**Best for**: AER processor, best performance
```yaml
detection:
  type: "HybridDetection"
  params:
    alpha: 0.5             # Weight: reconstruction vs prediction
    beta: 0.5              # Weight: forward vs backward prediction
```

**When to use**: With AERProcessor for SOTA performance
**Computational cost**: Low (combines pre-computed errors)
**Must use with**: AERProcessor

**Hyperparameter guidance**:
- `alpha`: Match the alpha from AERProcessor
- `beta`: 0.5 (balanced), 0.6 (favor forward), 0.4 (favor backward)

---

### AssociationDiscrepancyDetection (SOTA)
**Best for**: Anomaly Transformer, attention-based
```yaml
detection:
  type: "AssociationDiscrepancyDetection"
```

**When to use**: With AnomalyTransformerProcessor
**Computational cost**: Low (discrepancy already computed)
**Must use with**: AnomalyTransformerProcessor

---

### ClassificationBasedDetection
**Best for**: Supervised or semi-supervised scenarios
```yaml
detection:
  type: "ClassificationBasedDetection"
  params:
    classifier: "ocsvm"    # "ocsvm" or "iforest"
    nu: 0.1                # For OneClassSVM
```

**When to use**: Have some labeled anomalies for training
**Computational cost**: Medium

---

## Step 3: Scoring

### MaxPoolingScoring
**Best for**: Conservative detection (avoid missing anomalies)
```yaml
scoring:
  type: "MaxPoolingScoring"
```

**Effect**: Each point gets maximum score from all overlapping windows
**Tends to**: Higher recall, lower precision
**Use when**: Missing anomalies is costly

---

### AveragePoolingScoring
**Best for**: Balanced approach
```yaml
scoring:
  type: "AveragePoolingScoring"
```

**Effect**: Each point gets average score from all overlapping windows
**Tends to**: Balanced precision and recall
**Use when**: General purpose

---

### WeightedAverageScoring
**Best for**: Emphasize center of windows
```yaml
scoring:
  type: "WeightedAverageScoring"
```

**Effect**: Gaussian weights - window centers contribute more
**Tends to**: Smoother scores, better localization
**Use when**: Want smooth score transitions

---

### GaussianSmoothingScoring
**Best for**: Noisy scores, continuous anomalies
```yaml
scoring:
  type: "GaussianSmoothingScoring"
  params:
    sigma: 2.0             # Smoothing strength
```

**Effect**: Apply Gaussian smoothing after aggregation
**Tends to**: Very smooth scores, merge nearby anomalies
**Use when**: Scores are noisy or anomalies are continuous

**Hyperparameter guidance**:
- `sigma`: 1.0 (light smoothing), 2.0 (medium), 5.0 (heavy)

---

## Step 4: Post-Processing

### PercentileThreshold
**Best for**: Unsupervised, no validation labels
```yaml
postprocessing:
  threshold:
    type: "PercentileThreshold"
    params:
      percentile: 95.0     # 90-99.9
  min_anomaly_length: 3
  merge_gap: 5
```

**When to use**: No labeled validation data
**Typical values**: 95.0 (default), 99.0 (conservative)

---

### F1OptimalThreshold
**Best for**: Supervised, have validation labels
```yaml
postprocessing:
  threshold:
    type: "F1OptimalThreshold"
  min_anomaly_length: 3
  merge_gap: 5
```

**When to use**: Have validation set with labels
**Effect**: Finds threshold that maximizes F1 score
**Best practice**: Always use if you have labels

---

### StatisticalThreshold
**Best for**: Assume Gaussian normal distribution
```yaml
postprocessing:
  threshold:
    type: "StatisticalThreshold"
    params:
      k: 3.0               # Num std deviations
  min_anomaly_length: 3
  merge_gap: 5
```

**When to use**: Scores follow normal distribution
**Typical values**: k=3 (99.7% of normal data below threshold)

---

### Post-processing Parameters

**min_anomaly_length**: Remove anomaly segments shorter than this
- 1: Keep all detected anomalies (might be noisy)
- 3-5: Remove short spikes (recommended)
- 10+: Only keep substantial anomalies

**merge_gap**: Merge anomalies separated by this many points or less
- 0: Don't merge (keep distinct anomalies)
- 3-5: Merge nearby anomalies (recommended)
- 10+: Aggressive merging (might combine unrelated anomalies)

---

## Pre-configured Pipelines

### 1. Fast Baseline (KNN)
```yaml
data_processing: {type: "RawWindowProcessor", window_size: 100}
detection: {type: "DistanceBasedDetection", params: {k: 5}}
scoring: {type: "MaxPoolingScoring"}
postprocessing: {threshold: {type: "PercentileThreshold", params: {percentile: 95}}}
```
**Speed**: ⚡⚡⚡⚡⚡ | **Accuracy**: ★★★☆☆ | **GPU**: ❌

---

### 2. Interpretable Statistical
```yaml
data_processing: {type: "StatisticalFeatureProcessor", window_size: 100}
detection: {type: "ClassificationBasedDetection", params: {classifier: "iforest"}}
scoring: {type: "AveragePoolingScoring"}
postprocessing: {threshold: {type: "F1OptimalThreshold"}}
```
**Speed**: ⚡⚡⚡⚡☆ | **Accuracy**: ★★★☆☆ | **GPU**: ❌

---

### 3. AER SOTA Configuration
```yaml
data_processing:
  type: "AERProcessor"
  window_size: 100
  params: {hidden_dim: 128, num_layers: 2, alpha: 0.5}
detection:
  type: "HybridDetection"
  params: {alpha: 0.5, beta: 0.5}
scoring: {type: "WeightedAverageScoring"}
postprocessing: {threshold: {type: "F1OptimalThreshold"}, min_anomaly_length: 3, merge_gap: 5}
```
**Speed**: ⚡⚡☆☆☆ | **Accuracy**: ★★★★★ | **GPU**: ✅ Required

---

### 4. Anomaly Transformer SOTA
```yaml
data_processing:
  type: "AnomalyTransformerProcessor"
  window_size: 100
  params: {d_model: 512, n_heads: 8, n_layers: 3}
detection: {type: "AssociationDiscrepancyDetection"}
scoring: {type: "MaxPoolingScoring"}
postprocessing: {threshold: {type: "PercentileThreshold", params: {percentile: 95}}}
```
**Speed**: ⚡⚡⚡☆☆ | **Accuracy**: ★★★★★ | **GPU**: ✅ Required

---

## Decision Tree: Choose Your Pipeline

```
Start
  │
  ├─ Need SOTA performance?
  │    YES → Do you have GPU?
  │           YES → Do you prefer faster training?
  │                  YES → Anomaly Transformer (10 epochs)
  │                  NO  → AER (50 epochs, slightly better)
  │           NO  → Use baseline (KNN) or optimize on CPU
  │
  └─ NO → Do you need interpretability?
           YES → Statistical features + IsolationForest
           NO  → Do you have validation labels?
                  YES → KNN with F1OptimalThreshold
                  NO  → KNN with PercentileThreshold
```

---

## Performance vs Speed Trade-offs

| Configuration          | F1 Score | Training Time | Inference | GPU |
|------------------------|----------|---------------|-----------|-----|
| KNN Baseline           | 0.65     | 1 min         | 50 ms     | ❌  |
| Statistical + IForest  | 0.68     | 2 min         | 20 ms     | ❌  |
| LSTM-VAE              | 0.72     | 20 min        | 100 ms    | ⚠️  |
| **AER (SOTA)**        | **0.76** | 30 min        | 150 ms    | ✅  |
| Anomaly Transformer   | 0.75     | 10 min        | 120 ms    | ✅  |

---

## Common Pitfalls & Solutions

### Issue: Low F1 score on validation
**Solutions**:
1. Try F1OptimalThreshold instead of PercentileThreshold
2. Increase window_size (50 → 100 → 150)
3. Tune post-processing (min_anomaly_length, merge_gap)

### Issue: Training too slow
**Solutions**:
1. Reduce hidden_dim (256 → 128 → 64)
2. Reduce num_layers (3 → 2 → 1)
3. Use Anomaly Transformer instead of AER (faster)
4. Reduce batch_size if OOM

### Issue: High false positives
**Solutions**:
1. Increase threshold percentile (95 → 99)
2. Increase min_anomaly_length (3 → 5)
3. Use WeightedAverageScoring or GaussianSmoothing
4. Tune alpha in AER (favor reconstruction: 0.7)

### Issue: Missing anomalies (low recall)
**Solutions**:
1. Decrease threshold percentile (95 → 90)
2. Use MaxPoolingScoring instead of Average
3. Reduce min_anomaly_length (5 → 1)
4. Increase window_size to capture more context

### Issue: Out of memory
**Solutions**:
1. Reduce batch_size (32 → 16 → 8)
2. Reduce window_size (150 → 100 → 50)
3. Reduce d_model for Transformer (512 → 256)
4. Process data in smaller chunks

---

## Experimentation Workflow

1. **Start simple**: RawWindowProcessor + KNN + MaxPooling
2. **Add labels**: Switch to F1OptimalThreshold
3. **Upgrade model**: Try AER or Anomaly Transformer
4. **Tune hyperparameters**: Grid search window_size, hidden_dim, alpha
5. **Optimize post-processing**: Tune min_anomaly_length, merge_gap
6. **Ablation study**: Test different scoring methods
7. **Final selection**: Choose best config per dataset

---

## Example Ablation Study

Test these configurations in order:

```yaml
# Baseline
config_1: {processor: Raw, detection: KNN, scoring: Max, threshold: Percentile95}

# Better scoring
config_2: {processor: Raw, detection: KNN, scoring: Weighted, threshold: Percentile95}

# Better threshold
config_3: {processor: Raw, detection: KNN, scoring: Weighted, threshold: F1Optimal}

# SOTA processor
config_4: {processor: AER, detection: Hybrid, scoring: Weighted, threshold: F1Optimal}

# SOTA scoring
config_5: {processor: AER, detection: Hybrid, scoring: Gaussian(σ=2), threshold: F1Optimal}
```

Measure F1 improvement at each step to understand what matters most!

---

## References

- See `spec/architecture_overview.md` for design philosophy
- See `spec/phase1_infrastructure.md` for implementation details
- See `spec/phase2_sota_components.md` for SOTA methods
- See `README.md` for quick start guide
