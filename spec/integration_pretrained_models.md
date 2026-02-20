# Integration of Pre-trained Models (Inference Only)

## Overview

This document specifies how to integrate **pre-trained deep learning models** from Phase 2 (AER BiLSTM, Anomaly Transformer) into the Phase 3 foundation model approach. These models are used **inference-only** (no training) as optional evidence signals in the statistical evidence framework.

## Key Principle: Inference Only, No Training

**IMPORTANT**: Pre-trained models are used purely for inference (forward pass). No training, fine-tuning, or weight updates. This approach:
- ✅ Requires no GPU
- ✅ Works with existing pre-trained weights
- ✅ Provides additional evidence signal
- ✅ Can be disabled via configuration

## Supported Models

### 1. AER (Adversarial Encoder-Reconstructor) BiLSTM

**Architecture**: BiLSTM encoder + decoder with adversarial training

**Paper**: Zhang et al., "Time Series Anomaly Detection with Adversarial Reconstruction Networks" (2023)

**Pre-trained Weights**: If available from Phase 2 training or external sources

**Output**: Reconstruction error (anomaly score)

**Existing Implementation**: `src/models/aer.py`

### 2. Anomaly Transformer

**Architecture**: Transformer with anomaly attention mechanism

**Paper**: Xu et al., "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy", ICLR 2022

**Pre-trained Weights**: If available from Phase 2 training or external sources

**Output**: Anomaly attention weights (anomaly score)

**Existing Implementation**: `src/models/anomaly_transformer.py`

## Integration Architecture

### Position in Pipeline

Pre-trained models fit into **Step 2: Detection via Evidence Extraction** as **Category 5: Optional Pre-trained Model Evidence** in the 4-step pipeline (Phase 3).

**Pipeline Context:**
- **Step 1** (Preprocessing): Foundation model forecasting
- **Step 2** (Detection): **Statistical evidence extraction ← Pre-trained models used here**
- **Step 3** (Scoring): LLM reasoning
- **Step 4** (Post-Processing): Output parsing

```
STEP 2: DETECTION VIA EVIDENCE EXTRACTION
├── Category 1: Forecast-Based Evidence (MAE, MSE, quantiles, surprise)
├── Category 2: Statistical Tests (Z-score, Grubbs, CUSUM)
├── Category 3: Distribution-Based (KL divergence, Wasserstein)
├── Category 4: Pattern-Based (ACF, volatility, trend)
└── Category 5: Optional Pre-trained Models ← [AER, Transformer scores]
```

### Design Philosophy

Pre-trained models provide **complementary signals**:
- Foundation models (TimesFM, Chronos) → Forecast errors
- Statistical tests → Explicit thresholds
- Pre-trained deep models → **Learned patterns** from training

This creates a **hybrid scoring approach** combining zero-shot foundation models with learned representations.

## Implementation

### Loading Pre-trained Weights

```python
import torch
from src.models.aer import AER
from src.models.anomaly_transformer import AnomalyTransformer

class PretrainedModelWrapper:
    """Wrapper for inference-only use of pre-trained models."""

    def __init__(self, model_type, weights_path, device='cpu'):
        self.model_type = model_type
        self.weights_path = weights_path
        self.device = device
        self.model = None

        # Load model
        self._load_model()

    def _load_model(self):
        """Load pre-trained model weights."""

        if self.model_type == 'aer':
            # Initialize model architecture
            # (parameters should match training config)
            self.model = AER(
                window_size=100,
                input_dim=1,
                hidden_dim=64,
                num_layers=2
            )

        elif self.model_type == 'anomaly_transformer':
            self.model = AnomalyTransformer(
                window_size=100,
                input_dim=1,
                d_model=512,
                n_heads=8,
                e_layers=3
            )

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Load pre-trained weights
        if os.path.exists(self.weights_path):
            state_dict = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded pre-trained weights from {self.weights_path}")
        else:
            print(f"WARNING: Weights not found at {self.weights_path}. Model will use random initialization.")

        # Set to evaluation mode (disable dropout, batch norm updates, etc.)
        self.model.eval()
        self.model.to(self.device)

    def infer(self, time_series_window):
        """
        Run inference on a time series window.

        Args:
            time_series_window: numpy array of shape (window_size,) or (window_size, num_features)

        Returns:
            dict with anomaly score and metadata
        """

        # Convert to tensor
        if isinstance(time_series_window, np.ndarray):
            window_tensor = torch.FloatTensor(time_series_window)
        else:
            window_tensor = time_series_window

        # Add batch dimension if needed
        if window_tensor.dim() == 1:
            window_tensor = window_tensor.unsqueeze(0).unsqueeze(-1)  # (1, window_size, 1)
        elif window_tensor.dim() == 2:
            window_tensor = window_tensor.unsqueeze(0)  # (1, window_size, features)

        window_tensor = window_tensor.to(self.device)

        # Run inference (no gradient computation)
        with torch.no_grad():
            if self.model_type == 'aer':
                output = self._infer_aer(window_tensor)
            elif self.model_type == 'anomaly_transformer':
                output = self._infer_anomaly_transformer(window_tensor)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

        return output

    def _infer_aer(self, window_tensor):
        """Inference for AER model."""

        # AER returns reconstruction error
        reconstruction = self.model(window_tensor)
        reconstruction_error = torch.mean((window_tensor - reconstruction) ** 2, dim=-1)  # MSE per timestep

        # Aggregate to single score
        anomaly_score = torch.mean(reconstruction_error).item()
        max_error = torch.max(reconstruction_error).item()

        return {
            'aer_score': anomaly_score,
            'aer_max_error': max_error,
            'aer_error_per_timestep': reconstruction_error.squeeze().cpu().numpy()
        }

    def _infer_anomaly_transformer(self, window_tensor):
        """Inference for Anomaly Transformer."""

        # Anomaly Transformer returns anomaly attention
        output, anomaly_attention = self.model(window_tensor)

        # Aggregate attention to single score
        anomaly_score = torch.mean(anomaly_attention).item()
        max_attention = torch.max(anomaly_attention).item()

        return {
            'transformer_score': anomaly_score,
            'transformer_max_attention': max_attention,
            'transformer_attention_per_timestep': anomaly_attention.squeeze().cpu().numpy()
        }
```

### Integration with Evidence Extractor

```python
# In src/statistical_evidence/pretrained_models.py

class PretrainedModelEvidence:
    """Extract evidence from pre-trained models."""

    def __init__(self, config):
        self.config = config
        self.models = {}

        # Load models if enabled
        if config.get('use_aer', False):
            aer_weights = config.get('aer_weights_path')
            if aer_weights and os.path.exists(aer_weights):
                self.models['aer'] = PretrainedModelWrapper('aer', aer_weights)
            else:
                print("WARNING: AER enabled but weights not found. Skipping AER.")

        if config.get('use_transformer', False):
            transformer_weights = config.get('transformer_weights_path')
            if transformer_weights and os.path.exists(transformer_weights):
                self.models['transformer'] = PretrainedModelWrapper('anomaly_transformer', transformer_weights)
            else:
                print("WARNING: Transformer enabled but weights not found. Skipping Transformer.")

    def extract(self, time_series_window, historical_scores=None):
        """
        Extract evidence from pre-trained models.

        Args:
            time_series_window: numpy array
            historical_scores: dict with {'aer': [...], 'transformer': [...]} from training data

        Returns:
            dict with pre-trained model scores and anomaly flags
        """

        evidence = {}

        # AER
        if 'aer' in self.models:
            aer_output = self.models['aer'].infer(time_series_window)
            evidence.update(aer_output)

            # Threshold using historical scores
            if historical_scores and 'aer' in historical_scores:
                threshold = np.percentile(historical_scores['aer'], 95)
                evidence['aer_anomalous'] = aer_output['aer_score'] > threshold
                evidence['aer_threshold'] = threshold
                evidence['aer_percentile'] = percentileofscore(historical_scores['aer'], aer_output['aer_score'])

        # Anomaly Transformer
        if 'transformer' in self.models:
            transformer_output = self.models['transformer'].infer(time_series_window)
            evidence.update(transformer_output)

            # Threshold using historical scores
            if historical_scores and 'transformer' in historical_scores:
                threshold = np.percentile(historical_scores['transformer'], 95)
                evidence['transformer_anomalous'] = transformer_output['transformer_score'] > threshold
                evidence['transformer_threshold'] = threshold
                evidence['transformer_percentile'] = percentileofscore(historical_scores['transformer'], transformer_output['transformer_score'])

        return evidence
```

### Integration with Main Evidence Extractor

```python
# In src/statistical_evidence/evidence_extractor.py

class StatisticalEvidenceExtractor:
    def __init__(self, config):
        self.config = config
        # ... other initializations ...

        # Initialize pre-trained model evidence extractor
        if config.get('use_pretrained_models', False):
            pretrained_config = config.get('pretrained_models', {})
            self.pretrained_extractor = PretrainedModelEvidence(pretrained_config)
        else:
            self.pretrained_extractor = None

    def extract(self, train_data, test_data, forecast, forecast_samples=None):
        evidence = {}

        # Category 1-4: Statistical evidence
        # ... (forecast, statistical tests, distribution, pattern) ...

        # Category 5: Optional pre-trained models
        if self.pretrained_extractor:
            pretrained_evidence = self.pretrained_extractor.extract(
                test_data,
                historical_scores=self._get_historical_scores(train_data)
            )
            evidence.update(pretrained_evidence)

        return evidence

    def _get_historical_scores(self, train_data):
        """Compute scores on training data for threshold calibration."""

        if not self.pretrained_extractor:
            return None

        historical_scores = {'aer': [], 'transformer': []}

        # Slide over training data
        for i in range(len(train_data) - self.config['window_size']):
            window = train_data[i:i+self.config['window_size']]
            scores = self.pretrained_extractor.extract(window, historical_scores=None)

            if 'aer_score' in scores:
                historical_scores['aer'].append(scores['aer_score'])
            if 'transformer_score' in scores:
                historical_scores['transformer'].append(scores['transformer_score'])

        return historical_scores
```

## Configuration

### YAML Configuration

```yaml
statistical_evidence:
  use_pretrained_models: true

  pretrained_models:
    use_aer: true
    aer_weights_path: "pretrained/aer_yahoo_s5.pth"

    use_transformer: false  # Disabled (no weights available)
    transformer_weights_path: "pretrained/anomaly_transformer.pth"

  threshold_strategy: "percentile"  # percentile or fixed
  percentile_threshold: 95  # Use 95th percentile of training scores
```

### Python Configuration

```python
config = {
    'statistical_evidence': {
        'use_pretrained_models': True,
        'pretrained_models': {
            'use_aer': True,
            'aer_weights_path': 'pretrained/aer_weights.pth',
            'use_transformer': False
        },
        'window_size': 100,
        'threshold_strategy': 'percentile',
        'percentile_threshold': 95
    }
}
```

## Obtaining Pre-trained Weights

### Option 1: Phase 2 Training (If Available)

If you completed Phase 2 training before archiving, weights should be in:
```
best-tsad/
├── checkpoints/
│   ├── aer_model_best.pth
│   └── anomaly_transformer_best.pth
```

### Option 2: Download from External Sources

Check for publicly available pre-trained weights:
- **AER**: Search GitHub for AER implementations with weights
- **Anomaly Transformer**: Official repository may have pre-trained weights

### Option 3: Use Random Initialization (Fallback)

If no weights available, models will use random initialization. This is **not recommended** but allows testing the pipeline:
```python
# Model will initialize with random weights
# Performance will be poor, but code will run
wrapper = PretrainedModelWrapper('aer', weights_path='nonexistent.pth')
```

### Option 4: Train on Small Dataset (Future)

If GPU becomes available, train on a small representative dataset:
```python
# Minimal training for proof-of-concept
# (See archived/training_scripts/ for training code)
```

## Evidence Formatting for LLM

Pre-trained model scores are included in the evidence presented to the LLM:

```python
def format_pretrained_evidence(evidence):
    """Format pre-trained model evidence for LLM prompt."""

    if 'aer_score' not in evidence and 'transformer_score' not in evidence:
        return ""  # No pre-trained models used

    lines = ["### Pre-trained Model Scores (Learned Patterns):"]

    # AER
    if 'aer_score' in evidence:
        status = "ANOMALOUS" if evidence.get('aer_anomalous', False) else "NORMAL"
        percentile = evidence.get('aer_percentile', 0)
        lines.append(f"- AER BiLSTM Score: {evidence['aer_score']:.3f} "
                    f"({percentile:.1f}th percentile, {status})")

    # Anomaly Transformer
    if 'transformer_score' in evidence:
        status = "ANOMALOUS" if evidence.get('transformer_anomalous', False) else "NORMAL"
        percentile = evidence.get('transformer_percentile', 0)
        lines.append(f"- Anomaly Transformer Score: {evidence['transformer_score']:.3f} "
                    f"({percentile:.1f}th percentile, {status})")

    return "\n".join(lines)
```

Example in LLM prompt:
```
### Pre-trained Model Scores (Learned Patterns):
- AER BiLSTM Score: 0.823 (96.2th percentile, ANOMALOUS)
- Anomaly Transformer Score: 0.754 (92.1th percentile, ANOMALOUS)

These scores are based on deep learning models trained on similar time series data. High scores indicate patterns that deviate from learned normal behavior.
```

## Evaluation

### Contribution Analysis

Measure the added value of pre-trained models:

```python
def evaluate_pretrained_contribution(evidence_with_pretrained, evidence_without_pretrained, ground_truth):
    """Evaluate if pre-trained models improve detection."""

    # Run LLM with and without pre-trained evidence
    results_with = run_llm_agent(evidence_with_pretrained)
    results_without = run_llm_agent(evidence_without_pretrained)

    # Compare F1 scores
    f1_with = compute_f1(results_with, ground_truth)
    f1_without = compute_f1(results_without, ground_truth)

    improvement = f1_with - f1_without

    return {
        'f1_with_pretrained': f1_with,
        'f1_without_pretrained': f1_without,
        'improvement': improvement
    }
```

### Ablation Study

Test each component:
1. **Baseline**: Foundation model + statistical evidence only
2. **+AER**: Add AER scores
3. **+Transformer**: Add Transformer scores
4. **+Both**: Use both pre-trained models

Expected results:
- If pre-trained models add value → F1 improvement
- If no improvement → Can disable to simplify pipeline

## Advantages and Limitations

### Advantages

✅ **No training required**: Use existing weights
✅ **Complementary signal**: Captures learned patterns
✅ **Modular**: Easy to enable/disable
✅ **Hybrid approach**: Combines zero-shot + learned models
✅ **Validated models**: AER and Transformer are SOTA methods

### Limitations

❌ **Requires weights**: Need pre-trained weights (may not always be available)
❌ **Domain mismatch**: Weights trained on one dataset may not generalize
❌ **Extra inference cost**: Adds computational overhead (still CPU-friendly)
❌ **Black-box scores**: Less interpretable than statistical metrics

## Best Practices

1. **Start without pre-trained models**: Validate that foundation model + statistical evidence works first
2. **Add pre-trained models incrementally**: Enable AER, evaluate, then add Transformer
3. **Compare performance**: Use ablation studies to justify inclusion
4. **Document weight sources**: Track where weights came from (training run, external repo, etc.)
5. **Calibrate thresholds**: Use training data to set percentile thresholds
6. **Monitor inference time**: Ensure pre-trained models don't slow pipeline too much

## Future Extensions

### Transfer Learning

If GPU becomes available, fine-tune pre-trained models on new domains:
```python
# Minimal fine-tuning (few epochs, low learning rate)
pretrained_model = AER(...)
pretrained_model.load_state_dict(torch.load('pretrained/aer.pth'))

# Fine-tune on new dataset
optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=1e-5)
for epoch in range(5):  # Just a few epochs
    # ... training loop ...
```

### Ensemble with Foundation Models

Create ensemble scores combining foundation models and pre-trained models:
```python
def ensemble_score(forecast_error, aer_score, transformer_score, weights=[0.5, 0.25, 0.25]):
    """Weighted ensemble of multiple anomaly scores."""
    normalized_forecast_error = normalize(forecast_error)
    normalized_aer = normalize(aer_score)
    normalized_transformer = normalize(transformer_score)

    ensemble = (weights[0] * normalized_forecast_error +
                weights[1] * normalized_aer +
                weights[2] * normalized_transformer)

    return ensemble
```

### Attention Visualization

For Anomaly Transformer, visualize attention weights to understand what patterns it learned:
```python
def visualize_attention(time_series, attention_weights):
    """Plot time series with attention overlay."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(time_series, label='Time Series', color='blue')
    plt.fill_between(range(len(time_series)), attention_weights, alpha=0.3, color='red', label='Attention')
    plt.legend()
    plt.title('Anomaly Transformer Attention')
    plt.show()
```

## Comparison to Training-Based Approach

| Aspect | Phase 2 (Training) | Phase 3 (Inference Only) |
|--------|-------------------|------------------------|
| **GPU Required** | Yes (for training) | No |
| **Setup Time** | Hours (training) | Minutes (load weights) |
| **Adaptability** | Requires retraining | Use as-is or fine-tune |
| **Interpretability** | Black-box scores | Black-box scores (same) |
| **Performance** | Optimized for training data | Depends on weight source |
| **Role** | Primary detection method | Supplementary evidence signal |

## Summary

Pre-trained models from Phase 2 can be integrated as **optional evidence signals** in the Phase 3 pipeline:

1. **Load pre-trained weights** (inference only, no training)
2. **Run forward pass** to get anomaly scores
3. **Add scores to evidence dictionary** (Category 5)
4. **Present to LLM** as additional signal
5. **Evaluate contribution** via ablation studies

This hybrid approach combines:
- **Foundation models** (TimesFM, Chronos) → Zero-shot forecasting
- **Statistical evidence** → Explicit, interpretable signals
- **Pre-trained deep models** → Learned pattern representations
- **LLM reasoning** → Contextual understanding and explanation

---

**Status**: Specification complete, ready for implementation
**Last Updated**: 2026-02-17
**Dependencies**: PyTorch, pre-trained model weights (optional)
