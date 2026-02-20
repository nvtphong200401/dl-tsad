# Architecture Overview - Best TSAD (Time Series Anomaly Detection)

## Project Goal
Build a state-of-the-art time series anomaly detection system that achieves high performance with explainable outputs. The system uses a modular **4-step pipeline** that can operate in multiple modes:
- **Statistical Mode** (Phase 2): Traditional distance-based and reconstruction-based detection
- **Foundation + LLM Mode** (Phase 3): Zero-shot forecasting + evidence-based LLM reasoning

## 🆕 Phase 3: Foundation Model + LLM Approach (Current)

**Date**: 2026-02-17

**Major Architectural Shift**: Pivoted from training-based (Phase 2) to foundation model + LLM reasoning approach due to lack of GPU resources and to add explainability.

**Key Changes**:
- ❌ **Removed**: GPU training of deep learning models (archived)
- ✅ **Added**: Zero-shot foundation model forecasting (TimesFM, Chronos)
- ✅ **Added**: Multi-faceted statistical evidence framework (10+ metrics)
- ✅ **Added**: LLM reasoning layer for contextual understanding
- ✅ **Added**: RAG system for historical pattern retrieval
- ✅ **Preserved**: Core 4-step pipeline infrastructure, evaluation framework

**Critical Design Decision**: Phase 3 **enhances** the existing 4-step pipeline rather than extending it. Each step gains new capabilities while maintaining backward compatibility.

**Benefits**:
- GPU-free (runs on CPU or via API)
- Zero-shot generalization to new domains
- Explainable outputs (LLM reasoning with cited evidence)
- Faster experimentation (no training loops)
- Modular (can disable LLM, use statistical baseline)
- Backward compatible (can mix Phase 2 and Phase 3 components)

**See Also**:
- `foundation_model_llm_architecture.md` - Detailed Phase 3 architecture
- `MIGRATION_GUIDE.md` - Transition from Phase 2 to Phase 3
- `archived/README_ARCHIVED.md` - Explanation of archived training code

---

## Design Philosophy

### 1. Modularity
Each step in the pipeline is independent and replaceable. This allows:
- Testing different algorithms at each step
- Optimizing each component independently
- Easy comparison of different configurations
- Mix-and-match approach (e.g., foundation models with traditional scoring)

### 2. Formal Pipeline Structure

Based on research literature, every anomaly detection method follows this 4-step structure:

```
Step 1: Data Preprocessing → Step 2: Detection → Step 3: Scoring → Step 4: Post-Processing
```

**Phase 3 preserves this structure** by enhancing each step rather than adding new steps.

---

## 4-Step Pipeline Architecture

### Overview

The best-tsad system uses a modular 4-step pipeline that can operate in multiple modes:

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: DATA PREPROCESSING                             │
│  Phase 2: Windowing + Normalization                     │
│  Phase 3: + Foundation Model Forecasting                │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2: DETECTION                                       │
│  Phase 2: Distance-based (KNN), Reconstruction          │
│  Phase 3: Statistical Evidence Extraction (10+ metrics) │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 3: SCORING                                         │
│  Phase 2: Max/Average Pooling                           │
│  Phase 3: LLM Reasoning over Evidence                   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 4: POST-PROCESSING                                 │
│  Phase 2: Threshold + Filter/Merge                      │
│  Phase 3: + LLM Output Parsing + Explanations           │
└─────────────────────────────────────────────────────────┘
```

---

### Step 1: Data Preprocessing + Foundation Forecasting

**Input:** Raw time series `(T, D)` - T timesteps, D dimensions
**Output:** Processed windows `(N, W, D')` + Optional forecasts

#### Phase 2: Traditional Preprocessing

**Components:**
- Sliding window extraction (mandatory)
- Normalization (Z-score, Min-Max)
- Optional: Feature extraction, neural embeddings

**Examples:**
- Raw windows + normalization
- Statistical features (mean, std, skewness)
- Neural embeddings (AutoEncoder, LSTM encoder)

#### Phase 3: Enhanced with Foundation Models

**Additional Components:**
- **Foundation model forecasting** (TimesFM, Chronos)
- **Probabilistic predictions** (quantiles, uncertainty)
- **Ensemble predictions** (combine multiple models)

**Foundation Models:**
- **TimesFM** (Google): 200M-1.6B parameter decoder-only transformer
- **Chronos** (Amazon): T5-based probabilistic forecasting

**Why this fits Step 1:**
- Foundation models are **data transformers**, not detection algorithms
- Forecasting is a form of **data preprocessing** that enriches windows with predictions
- Keeps Step 1 focused on "preparing data for detection"

**Output:**
```python
{
    'windows': np.ndarray,        # (N, W, D') processed windows
    'forecasts': np.ndarray,      # (N, H) point forecasts (Phase 3)
    'quantiles': dict,            # {P10, P50, P90} (Phase 3)
    'train_distribution': dict    # Statistics for evidence (Phase 3)
}
```

---

### Step 2: Detection via Evidence Extraction

**Input:** Processed windows `(N, W, D')` + Optional forecasts
**Output:** Anomaly evidence (Phase 3) OR Sub-sequence scores `(N,)` (Phase 2)

#### Phase 2: Distance-Based and Reconstruction Methods

**Categories:**
- **Distance-based**: KNN, LOF, Isolation Forest
- **Reconstruction-based**: Compare original vs reconstructed (AutoEncoder)
- **Prediction-based**: Compare actual vs predicted (LSTM)
- **Hybrid**: AER (reconstruction + regression)
- **Classification-based**: OneClassSVM
- **Association-based**: Anomaly Transformer

**Output:** Single anomaly score per window `(N,)`

#### Phase 3: Statistical Evidence Extraction

**Components:**
- **Forecast-based evidence**: MAE, MSE, MAPE, quantile violations
- **Statistical tests**: Z-score, Grubbs test, CUSUM
- **Distribution metrics**: KL divergence, Wasserstein distance
- **Pattern analysis**: Autocorrelation breaks, volatility spikes, trend breaks
- **Optional**: Pre-trained model scores (AER, Transformer)

**Why this fits Step 2:**
- Evidence extraction **is** an anomaly detection method
- Both approaches identify anomalous patterns in windows
- Implements the same `DetectionMethod` interface
- Can operate in two modes:
  - **Statistical baseline**: Convert evidence → single score (backward compatible)
  - **LLM-ready**: Keep full evidence dict for Step 3 reasoning

**Output (Phase 3):**
```python
evidence = {
    'mae': 2.34, 'mae_percentile': 95.2, 'mae_anomalous': True,
    'z_score': 3.8, 'extreme_z_count': 2,
    'quantile_violations': {'above_p99': True, 'below_p01': False},
    'volatility_ratio': 5.2,
    'kl_divergence': 0.45,
    'autocorr_break': True,
    'aer_score': 0.82  # Optional
}
```

See `statistical_evidence_framework.md` for detailed specifications.

---

### Step 3: Scoring via Evidence Aggregation

**Input:** Sub-sequence scores `(N,)` OR Evidence dictionaries
**Output:** Anomaly scores + Optional explanations

#### Phase 2: Heuristic Pooling

**Purpose:** Convert window-level scores to point-level scores

**Methods:**
- **Max pooling**: Each point gets max score from overlapping windows
- **Average pooling**: Each point gets average score
- **Weighted average**: Gaussian or learned weights
- **Smoothing**: Gaussian filter, moving average

**Output:** Point-wise scores `(T,)`

#### Phase 3: LLM Reasoning as Intelligent Aggregation

**Purpose:** Aggregate statistical evidence via contextual understanding

**Process:**
1. Format evidence into structured prompt
2. Retrieve similar patterns via RAG (historical context)
3. LLM analyzes evidence + time series
4. Parse output: anomaly ranges + confidence + explanations

**LLM Models Supported:**
- GPT-4 Turbo (OpenAI)
- Gemini 1.5 Pro (Google)
- Claude 3 Opus (Anthropic)

**Why this fits Step 3:**
- Scoring is about **aggregating information to make decisions**
- Phase 2: Aggregate overlapping window scores → point scores (heuristic)
- Phase 3: Aggregate statistical evidence → explainable scores (intelligent)
- Both convert multiple signals into actionable scores
- LLM reasoning **replaces** heuristic pooling with contextual understanding

**Output (Phase 3):**
```json
{
  "anomalies": [
    {
      "start": 42, "end": 48,
      "confidence": 0.92,
      "reasoning": "Extreme Z-score (3.8) combined with quantile violation and 5.2x volatility spike. Pattern matches historical sensor failure case.",
      "evidence_cited": ["z_score", "quantile_violation", "volatility_spike"]
    }
  ],
  "mode": "llm_reasoning"
}
```

**Key Features:**
- Evidence-grounded (cites specific metrics)
- Contextual (uses RAG for similar patterns)
- Explainable (human-readable reasoning)

See `llm_reasoning_pipeline.md` for detailed specifications.

---

### Step 4: Post-Processing & Decision

**Input:** Point-wise scores `(T,)` OR Structured LLM output
**Output:** Binary anomaly labels `(T,)` + Evaluation metrics

#### Phase 2: Traditional Thresholding

**Components:**
- **Threshold determination**:
  - Percentile-based (e.g., 95th percentile)
  - F1-optimal (maximize F1 on validation set)
  - Statistical (mean + k*std)
  - Adaptive thresholding
- **Anomaly extraction**:
  - Point-wise vs interval detection
  - Filter short anomalies
  - Merge close anomalies

#### Phase 3: Enhanced with LLM Output Parsing

**Additional Components:**
- **Parse LLM structured outputs** (anomaly ranges, confidence, reasoning)
- **Extract evidence citations** for explainability
- **Convert anomaly ranges to binary labels**
- **Generate evaluation report with explanations**

**Preserved from Phase 2:**
- All evaluation metrics (F1, Precision, Recall, PA-F1, VUS-PR)
- Threshold-based extraction (for statistical baseline mode)
- Filter/merge operations

**Why this fits Step 4:**
- Post-processing remains focused on **final decision-making**
- Can handle both statistical scores and LLM outputs
- **Backward compatible**: If no LLM, use traditional thresholding

**Output:**
```python
{
    'predictions': np.ndarray,    # Binary labels (T,)
    'threshold': float,           # Used threshold (Phase 2)
    'explanations': list,         # Human-readable reasoning (Phase 3)
    'confidence': np.ndarray,     # Per-point confidence (Phase 3)
    'evidence_summary': dict      # Aggregated evidence (Phase 3)
}
```

---

## Phase 2 vs Phase 3 Comparison

| Aspect | Phase 2: Statistical | Phase 3: Foundation + LLM |
|--------|---------------------|---------------------------|
| **Step 1: Preprocessing** | Windowing + Normalization | + Foundation Model Forecasting |
| **Step 2: Detection** | Distance/Reconstruction (KNN, AutoEncoder) | Statistical Evidence Extraction (10+ metrics) |
| **Step 3: Scoring** | Heuristic Pooling (max/average) | LLM Reasoning (intelligent aggregation) |
| **Step 4: Post-Processing** | Threshold + Filter/Merge | + LLM Output Parsing + Explanations |
| **Training Required** | Yes (GPU-intensive) | No (zero-shot) |
| **Explainability** | None (black box scores) | Yes (cited evidence + reasoning) |
| **Inference Speed** | Fast (local) | Slower (API calls) |
| **Cost** | High (GPU training) | Low-Medium (API usage) |
| **Generalization** | Domain-specific (requires training data) | Zero-shot (pre-trained on 100B+ points) |
| **Modularity** | Can swap components within steps | Can mix Phase 2 and Phase 3 components |

**Key Insight**: Phase 3 enhances each step rather than adding new steps. This preserves the elegant 4-step architecture while adding zero-shot capabilities and explainability.

---

## Backward Compatibility

Phase 3 is designed to be **fully backward compatible** with Phase 2:

1. **Pure Statistical Mode**: Disable LLM reasoning, use evidence-based scores
2. **Hybrid Mode**: Use foundation forecasts with traditional scoring
3. **Full LLM Mode**: Use all Phase 3 components

**Configuration Example:**
```yaml
pipeline:
  step1:
    use_foundation_model: true  # Phase 3
  step2:
    use_evidence_extraction: true  # Phase 3
  step3:
    method: "max_pooling"  # Phase 2 (no LLM)
  step4:
    use_traditional_threshold: true  # Phase 2
```

---

## Technology Stack

### Phase 3: Foundation Model + LLM Stack (Current)

**Core Libraries:**
- **Python 3.9+**: Main language
- **NumPy**: Array operations
- **scipy**: Statistical operations
- **scikit-learn**: Metrics, statistical tests
- **statsmodels**: Time series analysis (ACF, trend detection)

**Foundation Models:**
- **timesfm**: Google's TimesFM for zero-shot forecasting
- **chronos-forecasting**: Amazon's Chronos for probabilistic predictions

**LLM Integration:**
- **openai**: GPT-4 API client
- **google-generativeai**: Gemini API client
- **anthropic**: Claude API client

**RAG System:**
- **chromadb**: Vector database for pattern retrieval
- **sentence-transformers**: Embedding models for evidence

**Configuration & Experiment Management:**
- **PyYAML**: Configuration files
- **python-dotenv**: Environment variables (API keys)

**Evaluation & Benchmarking:**
- **TSB-AD metrics**: VUS-PR implementation (preserved)
- **Custom metrics**: Point-adjusted F1 (preserved)

### Phase 2: Training-Based Stack (Archived but Supported)

**Deep Learning (Optional for inference):**
- **PyTorch**: Deep learning models (AER, Transformer)
- **torch.onnx**: Model export for production (optional)

**Note**: PyTorch is only needed if using pre-trained model scores as optional evidence in Step 2.
No training required in Phase 3.

---

## Project Structure

```
best-tsad/
├── spec/                          # Specifications
│   ├── architecture_overview.md       # This file
│   ├── foundation_model_llm_architecture.md
│   ├── statistical_evidence_framework.md
│   ├── llm_reasoning_pipeline.md
│   ├── rag_system_design.md
│   └── integration_pretrained_models.md
├── src/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── step1_data_processing.py    # Windowing + forecasting
│   │   ├── step2_detection.py          # Detection + evidence extraction
│   │   ├── step3_scoring.py            # Pooling + LLM reasoning
│   │   ├── step4_postprocessing.py     # Threshold + parsing
│   │   └── orchestrator.py             # Pipeline orchestrator
│   ├── foundation_models/
│   │   ├── timesfm_wrapper.py          # TimesFM integration
│   │   └── chronos_wrapper.py          # Chronos integration
│   ├── llm/
│   │   ├── reasoning_engine.py         # LLM reasoning logic
│   │   ├── prompt_templates.py         # Structured prompts
│   │   └── output_parser.py            # Parse LLM outputs
│   ├── rag/
│   │   ├── vector_store.py             # ChromaDB wrapper
│   │   └── retrieval_engine.py         # Pattern retrieval
│   ├── evidence/
│   │   ├── extractors.py               # Statistical evidence extraction
│   │   └── aggregators.py              # Evidence aggregation
│   ├── models/                         # Archived training-based models
│   │   ├── aer.py                      # AER BiLSTM (inference only)
│   │   └── anomaly_transformer.py      # Anomaly Transformer (inference only)
│   ├── evaluation/
│   │   ├── metrics.py                  # F1, PA-F1, VUS-PR
│   │   └── evaluator.py                # Evaluation orchestrator
│   ├── data/
│   │   ├── loader.py                   # Dataset loaders
│   │   └── normalization.py            # Normalization utilities
│   └── utils/
│       ├── config_factory.py           # Build pipeline from config
│       └── logging_utils.py
├── configs/
│   ├── datasets/
│   │   ├── smd.yaml
│   │   ├── msl.yaml
│   │   └── smap.yaml
│   ├── pipelines/
│   │   ├── phase3_full_llm.yaml        # Full Phase 3 pipeline
│   │   ├── phase3_statistical.yaml     # Phase 3 without LLM
│   │   ├── phase2_aer.yaml             # Legacy Phase 2 pipeline
│   │   └── hybrid_pipeline.yaml        # Mix Phase 2 + 3 components
│   └── experiments/
│       └── compare_all.yaml
├── data/                           # Downloaded datasets
│   ├── SMD/
│   ├── MSL/
│   └── SMAP/
├── tests/
│   ├── test_pipeline.py
│   ├── test_evidence_extraction.py
│   └── test_llm_reasoning.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_phase3_results_analysis.ipynb
├── experiments/
│   └── runs/                       # Experiment results
├── requirements.txt
├── setup.py
└── README.md
```

---

## Key Design Patterns

### 1. Abstract Base Classes
Each step defines an abstract interface that all implementations must follow:
- `DataProcessor` (Step 1) - handles both Phase 2 and Phase 3 preprocessing
- `DetectionMethod` (Step 2) - handles both scoring and evidence extraction
- `ScoringMethod` (Step 3) - handles both pooling and LLM reasoning
- `ThresholdDetermination` (Step 4) - handles both thresholding and parsing

### 2. Factory Pattern
Configuration files specify component types, and factories instantiate them:
```python
pipeline = build_pipeline_from_config(config)
```

### 3. Strategy Pattern
Each component can be swapped at runtime without changing the pipeline structure.

### 4. Pipeline Pattern
The orchestrator chains all 4 steps together and manages data flow.

---

## Evaluation Strategy

### Datasets
Standard benchmarks (preserved from Phase 2):
- **SMD** (Server Machine Dataset): 38-dimensional multivariate
- **MSL** (Mars Science Laboratory): NASA spacecraft telemetry
- **SMAP** (Soil Moisture Active Passive): NASA spacecraft telemetry
- **SWAT** (Secure Water Treatment): Industrial control system
- **PSM** (Pooled Server Metrics): Server metrics

### Metrics
- **F1 Score**: Standard precision-recall harmonic mean
- **PA-F1** (Point-Adjusted F1): Adjusted for continuous anomaly segments
- **VUS-PR**: Volume Under Surface for Precision-Recall curve
- **Precision & Recall**: Individual components
- **Latency**: p50, p99 inference time
- **Explanation Quality**: Human evaluation of LLM reasoning (Phase 3)

### Evaluation Protocol
1. Phase 2: Train on normal data (unsupervised), tune threshold
2. Phase 3: Zero-shot inference (no training), optional threshold tuning
3. Report metrics on test set
4. Cross-validation for robust estimates
5. Statistical significance testing

---

## Success Criteria

### Performance Targets (Phase 3)
- **Baseline**: F1 > 0.70 (Statistical evidence without LLM)
- **Target**: F1 > 0.75 (With LLM reasoning)
- **Stretch**: F1 > 0.80 (Optimized prompts + RAG)

### Engineering Targets
- Modular design: Can swap any component in < 10 lines of code
- Fast experimentation: Run new config in < 5 minutes
- Reproducible: Same config produces same results
- Well-tested: > 80% code coverage
- Explainable: 100% of predictions have human-readable reasoning

### Explainability Targets (Phase 3)
- Evidence citation: Every anomaly must cite specific metrics
- Human evaluation: > 80% of explanations rated as "clear and correct"
- RAG relevance: Retrieved patterns must have similarity > 0.7

---

## Summary

The best-tsad system uses a modular **4-step pipeline** that can operate in multiple modes. Phase 3 **enhances** the existing 4-step framework rather than extending it:

- **Step 1**: Preprocessing now includes foundation model forecasting
- **Step 2**: Detection now includes statistical evidence extraction
- **Step 3**: Scoring now includes LLM reasoning as intelligent aggregation
- **Step 4**: Post-processing now includes LLM output parsing

This design preserves the elegant, modular architecture while adding zero-shot forecasting, statistical grounding, and explainable LLM reasoning—without artificial complexity or breaking changes.

---

## References

### Key Papers
1. AER: Auto-Encoder with Regression (Wong et al., 2022)
2. Anomaly Transformer (Xu et al., ICLR 2022)
3. TranAD (Tuli et al., VLDB 2022)
4. TSB-AD Benchmark (NeurIPS 2024)
5. TimesFM: Google's Foundation Model for Time Series (2024)
6. Chronos: Learning the Language of Time Series (Amazon, 2024)

### Datasets
- SMD: https://github.com/NetManAIOps/OmniAnomaly
- MSL/SMAP: https://github.com/khundman/telemanom
- SWAT: https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/

### Metrics Implementation
- TSB-AD: https://github.com/TheDatumOrg/TSB-AD
- Point-Adjusted Metrics: From "Towards a Rigorous Evaluation of Time-Series Anomaly Detection" (AAAI 2022)
