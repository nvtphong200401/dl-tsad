# Foundation Model + LLM Architecture Specification

## Document Purpose

This is an **architecture specification** document that describes the foundation model + LLM approach for time series anomaly detection. This architecture is implemented across **Phase 2-4**:
- **Phase 2**: Core Implementation (foundation models, evidence extraction, LLM reasoning)
- **Phase 3**: Advanced Features (RAG system, optimization)
- **Phase 4**: Evaluation & Optimization

For **implementation guides**, see:
- `phase2_core_implementation.md` - Week-by-week implementation guide
- `phase3_advanced_features.md` - RAG and optimization guide
- `phase4_evaluation_optimization.md` - Evaluation guide

---

## Executive Summary

This specification describes a **GPU-free anomaly detection approach** that enhances the existing 4-step pipeline with:
1. **Pre-trained foundation forecasting models** (TimesFM, Chronos) for zero-shot predictive distributions
2. **Statistical evidence extraction** (10+ independent metrics) for quantitative anomaly signals
3. **LLM reasoning layer** (GPT-4, Gemini) for contextual understanding and explainable outputs
4. **Optional pre-trained models** (AER, Transformer) as additional scoring components (inference only)

**Critical Design Decision**: This approach **enhances the existing 4-step pipeline** rather than extending it to 5 steps. This preserves the elegant, modular architecture while adding zero-shot capabilities and explainability.

---

## Motivation

### Problems with Training-Based Approach (Archived)
- ❌ **GPU dependency**: Anomaly Transformer and AER require GPU for training
- ❌ **Resource bottleneck**: Training blocked research progress
- ❌ **Limited transferability**: Models trained on specific datasets may not generalize
- ❌ **Black-box predictions**: No explanation for anomaly decisions
- ❌ **Slow iteration**: Hyperparameter tuning requires multiple training runs

### Advantages of Foundation Model + LLM Approach (Current)
- ✅ **Zero-shot inference**: No training required, works on any time series
- ✅ **GPU-free**: Can run on CPU or via API
- ✅ **Probabilistic predictions**: Built-in uncertainty quantification
- ✅ **Explainable**: LLM provides reasoning and cites statistical evidence
- ✅ **Fast experimentation**: Immediate results, easy to iterate
- ✅ **Multi-faceted detection**: 10+ independent anomaly signals
- ✅ **Hybrid scoring**: Can incorporate pre-trained models (inference only)
- ✅ **Preserves 4-step architecture**: No breaking changes to pipeline structure

---

## Architecture Overview: Enhanced 4-Step Pipeline

This approach **enhances** the existing 4-step pipeline rather than extending it. Each step gains new capabilities while maintaining backward compatibility.

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: DATA PREPROCESSING + FOUNDATION FORECASTING        │
├─────────────────────────────────────────────────────────────┤
│ Baseline: Windowing + Normalization                        │
├─────────────────────────────────────────────────────────────┤
│ Enhancements (Phase 2):                                     │
│ • Foundation model forecasting (TimesFM, Chronos)          │
│ • Probabilistic predictions (quantiles, uncertainty)       │
│ • Ensemble predictions                                      │
│ • Training distribution statistics                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: DETECTION VIA EVIDENCE EXTRACTION                  │
├─────────────────────────────────────────────────────────────┤
│ Baseline: Distance-based (KNN), Reconstruction             │
├─────────────────────────────────────────────────────────────┤
│ Enhancements (Phase 2):                                     │
│ • Statistical evidence extraction (10+ metrics)            │
│   - Forecast errors (MAE, MSE, MAPE)                       │
│   - Statistical tests (Z-score, Grubbs, CUSUM)             │
│   - Distribution metrics (KL divergence, Wasserstein)      │
│   - Pattern analysis (autocorrelation, volatility, trends) │
│ • Optional: Pre-trained model scores (AER, Transformer)    │
│ • Output: Evidence dict OR anomaly scores                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: SCORING VIA EVIDENCE AGGREGATION                   │
├─────────────────────────────────────────────────────────────┤
│ Baseline: Heuristic Pooling (max/average)                  │
├─────────────────────────────────────────────────────────────┤
│ Enhancements (Phase 2-3):                                   │
│ • LLM reasoning over evidence + time series                │
│ • RAG context injection (historical patterns)              │
│ • Structured output: anomaly ranges + confidence + reasoning│
│ • Evidence citation for explainability                     │
│ • Fallback: Statistical aggregation (if LLM disabled)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: POST-PROCESSING & DECISION                         │
├─────────────────────────────────────────────────────────────┤
│ Baseline: Threshold + Filter/Merge                         │
├─────────────────────────────────────────────────────────────┤
│ Enhancements (Phase 2):                                     │
│ • Parse LLM structured outputs                             │
│ • Extract explanations and evidence citations              │
│ • Generate evaluation report with reasoning                │
│ • Fallback: Traditional thresholding (if LLM disabled)     │
└─────────────────────────────────────────────────────────────┘
```

### Key Conceptual Mappings

**Why Foundation Forecasting fits Step 1 (Preprocessing):**
- Foundation models are **data transformers**, not detection algorithms
- Forecasting enriches windows with predictions and uncertainty
- Keeps Step 1 focused on "preparing data for detection"

**Why Evidence Extraction fits Step 2 (Detection):**
- Evidence extraction **is** an anomaly detection method
- Both approaches identify anomalous patterns in windows
- Implements the same `DetectionMethod` interface
- Multiple metrics = multiple detection signals

**Why LLM Reasoning fits Step 3 (Scoring):**
- Scoring is about **aggregating information to make decisions**
- Phase 2: Aggregate overlapping window scores → point scores (heuristic)
- Phase 3: Aggregate statistical evidence → explainable scores (intelligent)
- LLM reasoning replaces heuristic pooling with contextual understanding

**Why LLM Output Parsing fits Step 4 (Post-Processing):**
- Post-processing handles final decision-making and formatting
- Can process both statistical scores and LLM outputs
- Backward compatible with traditional thresholding

---

## Step 1 Enhancement: Foundation Model Forecasting

### Purpose
Enrich data preprocessing with zero-shot forecasting to generate probabilistic predictions and uncertainty estimates.

### Foundation Models

#### TimesFM (Google Research)
- **Paper**: "A decoder-only foundation model for time-series forecasting" (ICML 2024)
- **Architecture**: Transformer decoder (200M-1.6B parameters)
- **Training**: Pre-trained on 100B+ time points from Google datasets
- **Strengths**: Fast inference, good on diverse domains, deterministic predictions
- **Limitations**: Single-point forecasts (no uncertainty by default)
- **Usage**: Zero-shot forecasting via HuggingFace or API

```python
from timesfm import TimesFM

model = TimesFM.from_pretrained("google/timesfm-1.0-200m")
forecast = model.forecast(
    time_series=train_data,
    horizon=forecast_horizon,
    context_length=context_length
)
```

#### Chronos (Amazon)
- **Paper**: "Chronos: Learning the Language of Time Series" (2024)
- **Architecture**: T5-based probabilistic forecasting
- **Training**: Pre-trained on diverse time series datasets
- **Strengths**: Probabilistic predictions (quantiles), uncertainty quantification
- **Limitations**: Slower than TimesFM
- **Usage**: HuggingFace `transformers` library

```python
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")
forecast = pipeline.predict(
    context=train_data,
    prediction_length=forecast_horizon,
    num_samples=100  # For quantile estimation
)
```

#### Ensemble Strategy
Combine both models for robust predictions:
- **Point forecast**: Average of TimesFM and Chronos median
- **Uncertainty**: Chronos quantiles (P10, P50, P90)
- **Confidence**: Consistency between models (low divergence = high confidence)

### Step 1 Output (Enhanced)
```python
{
    'windows': np.ndarray,           # (N, W, D') processed windows
    'forecasts': np.ndarray,         # (N, H) point forecasts
    'quantiles': dict,               # {P01, P10, P50, P90, P99}
    'train_distribution': dict,      # {mean, std, quantiles}
    'normalization_params': dict     # For denormalization
}
```

---

## Step 2 Enhancement: Statistical Evidence Extraction

### Purpose
Replace or augment distance-based detection with multi-faceted statistical evidence extraction.

### Evidence Categories

#### 1. Forecast-Based Evidence
Extract anomaly signals from forecast errors:

1. **Mean Absolute Error (MAE)**: `mean(|actual - forecast|)`
   - Threshold: 95th percentile of historical MAE
   - Signal: High MAE = poor prediction quality

2. **Quantile Violations**: `actual > P99 or actual < P01`
   - Binary signal: outside 99% confidence interval
   - Captures extreme deviations

3. **Surprise Score**: `-log P(actual | forecast_distribution)`
   - High surprise = low likelihood under model
   - Captures improbable events

#### 2. Statistical Test Evidence
Apply classical outlier detection tests:

4. **Z-Score**: `(x - mean) / std`
   - Threshold: |z| > 3 (3-sigma rule)
   - Standardized deviation from normal

5. **Grubbs Test**: Test for outliers in window
   - Statistic: `max(|x - mean|) / std`
   - Formally tests outlier hypothesis

6. **CUSUM**: Cumulative sum of deviations
   - Detects shifts in mean level
   - Sensitive to gradual drifts

#### 3. Distribution-Based Evidence
Compare train vs test distributions:

7. **KL Divergence**: `KL(P_train || P_test)`
   - Measures distribution shift
   - Detects changes in data generation process

8. **Wasserstein Distance**: Earth mover's distance
   - Robust to tail differences
   - Captures distributional changes

#### 4. Pattern-Based Evidence
Analyze time series patterns:

9. **Autocorrelation Break**: `|ACF(train) - ACF(test)|`
   - Detects periodicity changes
   - Captures seasonality shifts

10. **Volatility Spike**: `std(test) / std(train)`
    - Detects sudden variance changes
    - Captures instability

11. **Trend Break**: Piecewise linear fit deviation
    - Detects level shifts
    - Captures structural changes

#### 5. Optional: Pre-trained Model Scores
Use Phase 2 models as additional evidence:

12. **AER Score**: Use pre-trained AER model (inference only)
13. **Transformer Score**: Use pre-trained Anomaly Transformer (inference only)

### Key Design Principles

1. **Independence**: Each metric is computed independently
2. **Interpretability**: Each metric has a clear statistical meaning
3. **Flexibility**: Can enable/disable individual metrics
4. **Hybrid**: Can combine with traditional detection methods

### Step 2 Output (Enhanced)

**Mode 1: Evidence Dictionary (for LLM reasoning)**
```python
evidence = {
    # Forecast-based
    'mae': 2.34,
    'mae_percentile': 95.2,
    'mae_anomalous': True,
    'quantile_violations': {'above_p99': True, 'below_p01': False},
    'surprise_score': 4.2,

    # Statistical tests
    'z_score': 3.8,
    'extreme_z_count': 2,
    'grubbs_statistic': 3.5,
    'cusum_breach': True,

    # Distribution-based
    'kl_divergence': 0.45,
    'wasserstein_distance': 1.2,

    # Pattern-based
    'autocorr_break': True,
    'autocorr_delta': 0.6,
    'volatility_ratio': 5.2,
    'trend_break_score': 2.1,

    # Optional pre-trained models
    'aer_score': 0.82,
    'transformer_score': 0.79
}
```

**Mode 2: Aggregated Score (backward compatible)**
```python
# For statistical baseline (no LLM)
aggregated_score = weighted_average([
    evidence['mae_percentile'],
    evidence['z_score'] / 3.0,  # Normalize
    evidence['volatility_ratio'] / 10.0,
    # ...
])
```

See `statistical_evidence_framework.md` for detailed specifications.

---

## Step 3 Enhancement: LLM Reasoning as Intelligent Aggregation

### Purpose
Replace heuristic pooling with LLM reasoning to intelligently aggregate statistical evidence into explainable anomaly scores.

### Conceptual Shift

**Phase 2 (Heuristic Pooling):**
```
Multiple window scores → Max/Average → Point-wise scores
```

**Phase 3 (LLM Reasoning):**
```
Statistical evidence + Time series → LLM Analysis → Anomaly ranges + Explanations
```

Both approaches **aggregate multiple signals into actionable scores**, but Phase 3 uses contextual understanding instead of heuristics.

### Supported LLM Models

- **GPT-4 Turbo (OpenAI)**: Best reasoning, expensive
- **Gemini 1.5 Pro (Google)**: Long context (1M tokens), cost-effective
- **Claude 3 Opus (Anthropic)**: Strong reasoning, moderate cost
- **Gemini 2.0 Flash (Google)**: Fast, cheap, good for prototyping

### Prompt Engineering Strategy

1. **Role definition**: "You are an expert time series analyst..."
2. **Data presentation**: Show time series values + timestamps
3. **Evidence summary**: List all statistical signals with thresholds
4. **Historical context**: Inject similar patterns via RAG
5. **Structured output**: Request JSON with anomaly ranges, confidence, reasoning

### Example Prompt Template

```
You are an expert time series analyst. Analyze this window for anomalies.

Time Series (Window 42):
[t0: 1.2, t1: 1.3, t2: 1.5, ..., t99: 5.8]

Statistical Evidence:
• Forecast Error (MAE): 2.34 (HIGH - 95th percentile)
• Z-Score: 3.8 (EXTREME - exceeds 3-sigma threshold)
• Quantile Violation: TRUE (actual value > P99 forecast quantile)
• Volatility Spike: 5.2x baseline (SEVERE - sudden variance increase)
• Autocorrelation Break: 0.8 → 0.2 (periodicity loss)
• KL Divergence: 0.45 (moderate distribution shift)
• AER Model Score: 0.82 (high anomaly likelihood)

Historical Context (RAG):
• Similar pattern (similarity: 0.89) in Case #127: Sensor malfunction
• Similar pattern (similarity: 0.82) in Case #204: Network congestion

Question: Are there anomalies in this window?
Provide:
1. Anomaly ranges (timestep start-end)
2. Confidence score (0-1)
3. Reasoning (cite specific evidence)
4. Evidence used in decision

Output format: JSON
```

### LLM Output Parsing

Expected LLM output format:
```json
{
  "anomalies": [
    {
      "start": 42,
      "end": 48,
      "confidence": 0.92,
      "reasoning": "Extreme Z-score (3.8) combined with quantile violation (actual > P99) and severe volatility spike (5.2x baseline). The autocorrelation break suggests a fundamental pattern change. This matches historical Case #127 (sensor malfunction) with 89% similarity.",
      "evidence_cited": [
        "z_score",
        "quantile_violation",
        "volatility_spike",
        "autocorr_break",
        "historical_case_127"
      ]
    }
  ],
  "overall_assessment": "High confidence anomaly detected due to convergent statistical signals and strong historical pattern match.",
  "uncertainty_factors": [
    "Moderate KL divergence suggests some normal variation",
    "Only one strong historical match"
  ]
}
```

### Fallback: Statistical Aggregation

If LLM is disabled (cost-sensitive mode):
```python
# Simple weighted aggregation of evidence metrics
score = (
    0.3 * normalize(evidence['mae_percentile']) +
    0.2 * normalize(evidence['z_score']) +
    0.2 * normalize(evidence['volatility_ratio']) +
    0.15 * normalize(evidence['kl_divergence']) +
    0.15 * (evidence['aer_score'] if available else 0)
)
```

### Step 3 Output (Enhanced)

```python
{
    'anomaly_scores': np.ndarray,      # (N,) or (T,) depending on mode
    'anomaly_ranges': list,            # [(start, end), ...] from LLM
    'confidence': np.ndarray,          # (N,) per-window confidence
    'reasoning': list,                 # Human-readable explanations
    'evidence_cited': list,            # Which metrics were used
    'mode': 'llm_reasoning'            # or 'statistical_baseline'
}
```

See `llm_reasoning_pipeline.md` for detailed specifications.

---

## Step 4 Enhancement: LLM Output Parsing & Explainability

### Purpose
Extend post-processing to handle LLM structured outputs and generate explainable evaluation reports.

### Components

#### 1. Parse LLM Outputs
```python
def parse_llm_output(llm_response):
    """Convert LLM JSON to binary labels"""
    parsed = json.loads(llm_response)

    # Extract anomaly ranges
    anomaly_ranges = [
        (a['start'], a['end'])
        for a in parsed['anomalies']
    ]

    # Convert to binary labels
    binary_labels = np.zeros(len(time_series))
    for start, end in anomaly_ranges:
        binary_labels[start:end+1] = 1

    return binary_labels, parsed
```

#### 2. Extract Explanations
```python
explanations = {
    'anomaly_ranges': parsed['anomalies'],
    'overall_assessment': parsed['overall_assessment'],
    'evidence_summary': aggregate_evidence(parsed['evidence_cited']),
    'confidence_distribution': np.array([a['confidence'] for a in parsed['anomalies']])
}
```

#### 3. Traditional Post-Processing (Preserved)
All Phase 2 post-processing operations still work:
- Filter short anomalies
- Merge close anomalies
- Adaptive thresholding (if statistical mode)

#### 4. Generate Evaluation Report
```python
report = {
    'metrics': {
        'F1': f1_score,
        'Precision': precision,
        'Recall': recall,
        'PA-F1': pa_f1,
        'VUS-PR': vus_pr
    },
    'explanations': explanations,
    'evidence_importance': rank_evidence_by_usage(parsed)
}
```

### Step 4 Output (Final)

```python
{
    'predictions': np.ndarray,           # (T,) binary labels
    'threshold': float,                  # Used threshold (if statistical mode)
    'explanations': dict,                # Human-readable reasoning
    'confidence': np.ndarray,            # (T,) per-point confidence
    'evidence_summary': dict,            # Aggregated evidence statistics
    'evaluation_metrics': dict,          # F1, PA-F1, VUS-PR, etc.
    'mode': 'llm_reasoning'              # or 'statistical_baseline'
}
```

---

## RAG System for Historical Patterns

### Purpose
Provide LLM with relevant historical examples to improve reasoning consistency and leverage past knowledge.

### Implementation

#### 1. Vector Database
Store embeddings of historical patterns:
```python
pattern_entry = {
    'time_series': window_data,
    'evidence': evidence_dict,
    'label': anomaly_label,
    'reasoning': human_annotation,
    'embedding': embed(evidence_dict)  # Sentence-Transformers
}
```

#### 2. Retrieval
Query for similar patterns:
```python
# Query by evidence profile
query_embedding = embed(current_evidence)
similar_patterns = vector_db.query(
    query_embedding,
    top_k=3,
    threshold=0.7  # Minimum similarity
)
```

#### 3. Context Injection
Add to LLM prompt:
```
Historical Context:
• Pattern #127 (similarity: 0.89): Sensor malfunction
  Evidence: Z-score=3.9, volatility_ratio=5.1x
• Pattern #204 (similarity: 0.82): Network congestion
  Evidence: Z-score=3.2, autocorr_break=True
```

#### 4. Continuous Learning
After evaluation, add new patterns:
```python
vector_db.add({
    'time_series': window,
    'evidence': evidence,
    'label': ground_truth,
    'reasoning': llm_output['reasoning']
})
```

See `spec/rag_system_design.md` for detailed design.

---

## Integration with Pre-trained Models

Pre-trained AER and Anomaly Transformer models from Phase 2 can be used as **optional evidence signals** in Step 2:

```python
# Load pre-trained model (inference only, no training)
aer_model = AER(window_size=100, input_dim=1)
aer_model.load_state_dict(torch.load('pretrained/aer_weights.pth'))
aer_model.eval()

# Run inference
with torch.no_grad():
    aer_score = aer_model(window).item()

# Add to evidence dictionary
evidence['aer_score'] = aer_score  # Just another signal
```

**Key Point**: Pre-trained models are treated as **evidence generators**, not standalone detection methods. They integrate seamlessly into Step 2.

See `spec/integration_pretrained_models.md` for details.

---

## Comparison: Phase 2 vs Phase 3 (Within 4-Step Framework)

| Aspect | Phase 2: Statistical | Phase 3: Foundation + LLM |
|--------|---------------------|---------------------------|
| **Pipeline Structure** | 4 steps | 4 steps (enhanced) |
| **Step 1: Preprocessing** | Windowing + Normalization | + Foundation Forecasting |
| **Step 2: Detection** | Distance/Reconstruction (KNN, AutoEncoder) | Statistical Evidence (10+ metrics) |
| **Step 3: Scoring** | Heuristic Pooling (max/average) | LLM Reasoning (intelligent) |
| **Step 4: Post-Processing** | Threshold + Filter/Merge | + LLM Parsing + Explanations |
| **Training Required** | Yes (GPU-intensive) | No (zero-shot) |
| **Explainability** | None (black box) | Yes (cited evidence + reasoning) |
| **Inference Speed** | Fast (local) | Slower (API calls) |
| **Cost** | High (GPU training) | Low-Medium (API usage) |
| **Generalization** | Domain-specific | Zero-shot (pre-trained) |
| **Modularity** | Swap components within steps | Mix Phase 2 & 3 components |

**Key Insight**: Phase 3 enhances each step rather than adding new steps. This preserves the elegant 4-step architecture.

---

## Evaluation Strategy

### Metrics (Unchanged from Phase 2)
- Point-based: Precision, Recall, F1-Score
- Event-based: PA-Precision, PA-Recall, PA-F1
- Volume-based: VUS-ROC, VUS-PR

### Baselines to Compare
1. **Pure statistical** (no LLM): Threshold on evidence metrics
2. **Pure LLM** (no foundation models): Direct time series to LLM
3. **Training-based** (Phase 2): Pre-trained AER/Transformer scores
4. **Foundation + LLM** (Phase 3): Full enhanced pipeline
5. **AnomLLM benchmark**: GPT-4 with sliding window

### Ablation Studies
- Effect of each evidence metric
- Impact of RAG system
- Comparison of LLM models (GPT-4 vs Gemini vs Claude)
- Ensemble vs single foundation model
- Statistical baseline vs LLM reasoning

---

## Implementation Roadmap

### Phase 3.1: Step 1 Enhancement - Foundation Model Integration (Week 1-2)
- [ ] Install TimesFM and Chronos libraries
- [ ] Create wrapper classes in `src/foundation_models/`
- [ ] Implement ensemble forecasting
- [ ] Update `step1_data_processing.py` to output forecasts
- [ ] Test on sample datasets

### Phase 3.2: Step 2 Enhancement - Statistical Evidence Extraction (Week 2-3)
- [ ] Create `src/evidence/extractors.py` module
- [ ] Implement 10+ evidence metrics
- [ ] Update `step2_detection.py` to support evidence mode
- [ ] Integrate with pre-trained models (optional)
- [ ] Validate evidence quality

### Phase 3.3: Step 3 Enhancement - LLM Reasoning Layer (Week 3-4)
- [ ] Create `src/llm/reasoning_engine.py`
- [ ] Implement evidence-based prompt builder
- [ ] Update `step3_scoring.py` to support LLM reasoning mode
- [ ] Implement output parser
- [ ] Test on synthetic data

### Phase 3.4: RAG System (Week 4-5)
- [ ] Set up vector database (ChromaDB or FAISS)
- [ ] Implement pattern retrieval in `src/rag/`
- [ ] Populate with known patterns
- [ ] Integrate with LLM reasoning engine

### Phase 3.5: Step 4 Enhancement - LLM Output Parsing (Week 5)
- [ ] Update `step4_postprocessing.py` to parse LLM outputs
- [ ] Implement explanation extraction
- [ ] Generate enhanced evaluation reports

### Phase 3.6: Pipeline Integration & Evaluation (Week 6)
- [ ] Update orchestrator for Phase 3 components
- [ ] Configuration system for new modes
- [ ] Comprehensive evaluation on benchmark datasets
- [ ] Compare to Phase 2 baseline

---

## Research Contributions

This approach enables several novel research contributions:

1. **Enhanced 4-Step Architecture**: First work to enhance traditional pipeline with foundation models + LLM reasoning without changing pipeline structure
2. **Multi-Faceted Evidence**: Framework for extracting 10+ independent anomaly signals
3. **Explainable Detection**: Anomaly decisions with cited evidence and reasoning
4. **Zero-Shot Generalization**: Works on unseen domains without training
5. **RAG for Time Series**: Novel application of retrieval-augmented generation to TSAD
6. **Backward Compatible**: Can mix Phase 2 and Phase 3 components

---

## Configuration Examples

### Full Phase 3 Pipeline (LLM Mode)
```yaml
pipeline:
  step1:
    use_foundation_model: true
    models: ['timesfm', 'chronos']
  step2:
    method: 'evidence_extraction'
    metrics: ['mae', 'z_score', 'volatility', 'kl_div']
  step3:
    method: 'llm_reasoning'
    model: 'gpt-4'
    use_rag: true
  step4:
    parse_llm_output: true
    extract_explanations: true
```

### Phase 3 Statistical Baseline (No LLM)
```yaml
pipeline:
  step1:
    use_foundation_model: true
  step2:
    method: 'evidence_extraction'
  step3:
    method: 'statistical_aggregation'  # No LLM
  step4:
    use_traditional_threshold: true
```

### Hybrid: Phase 3 Step 1-2 + Phase 2 Step 3-4
```yaml
pipeline:
  step1:
    use_foundation_model: true  # Phase 3
  step2:
    method: 'evidence_extraction'  # Phase 3
  step3:
    method: 'max_pooling'  # Phase 2
  step4:
    use_traditional_threshold: true  # Phase 2
```

---

## References

### Foundation Models
- **TimesFM**: Das et al., "A decoder-only foundation model for time-series forecasting", ICML 2024
- **Chronos**: Ansari et al., "Chronos: Learning the Language of Time Series", arXiv 2024

### LLM for Time Series
- **LLM-TSAD**: Our reference implementation (see `../LLM-TSAD/`)
- **AnomLLM**: Benchmark for LLM-based anomaly detection (see `../AnomLLM/`)

### Anomaly Detection
- **Anomaly Transformer**: Xu et al., ICLR 2022
- **AER**: Zhang et al., 2023
- **TSB-AD Benchmark**: NeurIPS 2024

---

## Summary

Phase 3 **enhances the existing 4-step pipeline** rather than extending it to 5 steps:

- **Step 1**: Preprocessing now includes foundation model forecasting
- **Step 2**: Detection now includes statistical evidence extraction
- **Step 3**: Scoring now includes LLM reasoning as intelligent aggregation
- **Step 4**: Post-processing now includes LLM output parsing

This design preserves the elegant, modular architecture while adding zero-shot forecasting, statistical grounding, and explainable LLM reasoning—without artificial complexity or breaking changes.

---

**Status**: Specification complete, ready for implementation
**Last Updated**: 2026-02-17
**Author**: best-tsad team
