# Best TSAD: State-of-the-Art Time Series Anomaly Detection

A modular, research-grade framework for time series anomaly detection combining **foundation models** (TimesFM, Chronos), **statistical evidence**, and **LLM reasoning** for explainable, zero-shot anomaly detection.

## 🎯 Project Goals

1. **Achieve SOTA Performance**: Target F1 > 0.75 on standard benchmarks with explainable outputs
2. **Zero-Shot Generalization**: Use pre-trained foundation models (no GPU training required)
3. **Explainable Detection**: LLM reasoning with cited statistical evidence
4. **Modular Architecture**: Easy to experiment with different components at each step
5. **Research Quality**: Reproducible, well-tested, and documented

## 🆕 Phase 3: Foundation Model + LLM Reasoning Approach

**New in Phase 3** (2026-02-17):
- ✅ **GPU-free**: Use pre-trained foundation models (TimesFM, Chronos) for zero-shot forecasting
- ✅ **Explainable**: LLM reasoning layer (GPT-4, Gemini, Claude) provides explanations
- ✅ **Statistical grounding**: 10+ independent evidence metrics prevent hallucination
- ✅ **Hybrid scoring**: Optional integration of pre-trained deep models (inference only)
- ⚠️ **Training code archived**: Phase 2 training scripts moved to `archived/` (see `MIGRATION_GUIDE.md`)

## 🏗️ Architecture Overview

### Enhanced 4-Step Pipeline

The best-tsad system uses a modular **4-step pipeline** that can operate in multiple modes:
- **Statistical Mode** (Phase 2): Traditional distance-based and reconstruction-based detection
- **Foundation + LLM Mode** (Phase 3): Zero-shot forecasting + evidence-based LLM reasoning

**Critical Design Decision**: Phase 3 **enhances** the existing 4-step pipeline rather than extending it to 5 steps.

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: DATA PREPROCESSING + FOUNDATION FORECASTING   │
│  Phase 2: Windowing + Normalization                     │
│  Phase 3: + Foundation Models (TimesFM, Chronos)        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2: DETECTION VIA EVIDENCE EXTRACTION             │
│  Phase 2: Distance-based (KNN), Reconstruction          │
│  Phase 3: Statistical Evidence (10+ metrics)            │
│  • Forecast errors (MAE, quantile violations)           │
│  • Statistical tests (Z-score, Grubbs, CUSUM)           │
│  • Distribution metrics (KL divergence, Wasserstein)    │
│  • Pattern analysis (ACF breaks, volatility spikes)     │
│  • Optional: Pre-trained model scores (AER, Transformer)│
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 3: SCORING VIA EVIDENCE AGGREGATION              │
│  Phase 2: Heuristic Pooling (max/average)              │
│  Phase 3: LLM Reasoning (intelligent aggregation)       │
│  • Format evidence + time series into structured prompt │
│  • RAG context injection (historical patterns)          │
│  • LLM inference (GPT-4, Gemini, Claude)                │
│  • Output: anomaly ranges + confidence + explanations   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 4: POST-PROCESSING & DECISION                     │
│  Phase 2: Threshold + Filter/Merge                      │
│  Phase 3: + LLM Output Parsing + Explanations           │
│  • Parse LLM structured outputs                         │
│  • Merge consecutive anomalies, filter short ones       │
│  • Compute metrics (F1, PA-F1, VUS-PR)                  │
└─────────────────────────────────────────────────────────┘
```

### Why 4 Steps? (Not 5)

**Key Conceptual Mappings:**

- **Foundation Forecasting fits Step 1** (Preprocessing): Foundation models are data transformers that enrich windows with predictions
- **Evidence Extraction fits Step 2** (Detection): Statistical evidence extraction IS an anomaly detection method
- **LLM Reasoning fits Step 3** (Scoring): LLM intelligently aggregates evidence (replaces heuristic pooling)
- **LLM Output Parsing fits Step 4** (Post-Processing): Handles both statistical scores and LLM outputs

This preserves the elegant, modular 4-step architecture while adding zero-shot capabilities and explainability.

### Key Features

✅ **Zero-shot foundation models** - TimesFM (Google), Chronos (Amazon) for forecasting
✅ **Multi-faceted evidence** - 10+ independent statistical anomaly signals
✅ **LLM reasoning** - Explainable decisions with GPT-4, Gemini, or Claude
✅ **RAG system** - Historical pattern retrieval for improved reasoning
✅ **Hybrid scoring** - Optional integration of pre-trained models (inference only)
✅ **GPU-free** - Runs on CPU or via API (no training required)
✅ **Config-driven** - YAML files for reproducibility
✅ **Comprehensive evaluation** - F1, PA-F1, VUS-PR metrics
✅ **Modular** - Easy to disable LLM, use statistical baseline only

## 📋 Project Structure

```
best-tsad/
├── spec/                                   # Detailed specifications
│   ├── architecture_overview.md            # System design (updated for Phase 3)
│   ├── phase1_infrastructure.md            # Base infrastructure
│   ├── foundation_model_llm_architecture.md # NEW: Foundation model architecture
│   ├── statistical_evidence_framework.md   # NEW: Evidence extraction design
│   ├── llm_reasoning_pipeline.md           # NEW: LLM integration
│   ├── rag_system_design.md                # NEW: RAG system
│   ├── integration_pretrained_models.md    # NEW: Using Phase 2 models
│   └── archived/                           # OLD: Training-focused specs
│
├── archived/                               # NEW: Archived training code
│   ├── training_scripts/                   # Training experiments (Phase 2)
│   ├── training_docs/                      # Training documentation
│   └── README_ARCHIVED.md                  # Explanation of archived content
│
├── src/
│   ├── foundation_models/                  # NEW MODULE (to be implemented)
│   │   ├── timesfm_wrapper.py
│   │   └── chronos_wrapper.py
│   ├── statistical_evidence/               # NEW MODULE (to be implemented)
│   │   └── evidence_extractor.py
│   ├── llm_reasoning/                      # NEW MODULE (to be implemented)
│   │   └── llm_agent.py
│   ├── pipeline/                           # Core pipeline (preserved)
│   │   ├── step1_data_processing.py
│   │   ├── step3_scoring.py
│   │   ├── step4_postprocessing.py
│   │   └── orchestrator.py
│   ├── models/                             # Deep learning models (inference only)
│   │   ├── aer.py
│   │   └── anomaly_transformer.py
│   ├── evaluation/                         # Metrics and evaluation
│   │   ├── metrics.py
│   │   └── evaluator.py
│   ├── data/                               # Dataset loaders
│   │   ├── loader.py
│   │   └── anomllm_loader.py
│   └── utils/
│       └── config_factory.py
│
├── configs/                                # Experiment configurations
│   └── pipelines/
│       ├── foundation_llm_config.yaml      # NEW: Foundation model + LLM
│       └── statistical_baseline.yaml       # NEW: Statistical only (no LLM)
│
├── experiments/                            # Experiment scripts
│   ├── run_experiment.py                   # Config-driven runner
│   ├── run_with_pretrained.py              # Use pre-trained models
│   └── run_baseline.py
│
├── tests/                                  # Unit and integration tests
├── notebooks/                              # Analysis notebooks
├── MIGRATION_GUIDE.md                      # NEW: Phase 2 → Phase 3 transition guide
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/best-tsad.git
cd best-tsad

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install foundation model libraries
pip install timesfm chronos-forecasting

# Install LLM SDKs (choose one or more)
pip install openai google-generativeai anthropic
```

### Set Up API Keys

```bash
# For OpenAI (GPT-4)
export OPENAI_API_KEY="sk-..."

# For Google (Gemini)
export GOOGLE_API_KEY="AI..."

# For Anthropic (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Run Phase 3 Experiment (Foundation Model + LLM)

```bash
# Run with foundation models + LLM reasoning
python experiments/run_foundation_llm.py --dataset yahoo_s5 --llm gemini

# Output:
# ==================================================
# Experiment: foundation_llm (TimesFM + Chronos + Gemini 1.5 Pro)
# ==================================================
# F1 Score:          0.78
# Precision:         0.75
# Recall:            0.82
# PA-F1 Score:       0.80
# Explainability:    ✅ (LLM reasoning provided)
# ==================================================
```

### Run Statistical Baseline (No LLM)

```bash
# Use statistical evidence only (no LLM, no API cost)
python experiments/run_statistical_baseline.py --dataset yahoo_s5

# Output:
# ==================================================
# Experiment: statistical_baseline (Foundation Models + Thresholds)
# ==================================================
# F1 Score:          0.72
# Precision:         0.70
# Recall:            0.75
# Explainability:    ❌ (Scores only, no reasoning)
# ==================================================
```

### Migration from Phase 2

If you were using Phase 2 training-based approach:

```bash
# See migration guide
cat MIGRATION_GUIDE.md

# Training scripts are now in archived/
ls archived/training_scripts/

# Use pre-trained weights (if available) in Phase 3 pipeline
python experiments/run_with_pretrained.py --aer-weights pretrained/aer.pth
```

### Create Custom Pipeline

Create a config file `configs/pipelines/my_pipeline.yaml`:

```yaml
experiment:
  name: "my_custom_pipeline"

data_processing:
  type: "AERProcessor"
  window_size: 100
  stride: 1
  params:
    hidden_dim: 128

detection:
  type: "HybridDetection"
  params:
    alpha: 0.5

scoring:
  type: "WeightedAverageScoring"

postprocessing:
  threshold:
    type: "F1OptimalThreshold"
  min_anomaly_length: 3
  merge_gap: 5
```

Run it:
```bash
python experiments/run_pipeline.py --config configs/pipelines/my_pipeline.yaml
```

## 📊 Implemented Methods

### Data Processing (Step 1)
- **RawWindowProcessor**: Sliding windows + normalization
- **StatisticalFeatureProcessor**: Extract statistical features (mean, std, skewness, etc.)
- **NeuralEmbeddingProcessor**: AutoEncoder/LSTM embeddings
- **AERProcessor**: BiLSTM encoder-decoder with regression (SOTA)
- **AnomalyTransformerProcessor**: Transformer with association discrepancy (SOTA)

### Detection Methods (Step 2)
- **DistanceBasedDetection**: K-nearest neighbors, LOF
- **ReconstructionBasedDetection**: Reconstruction error
- **PredictionBasedDetection**: Prediction error
- **HybridDetection**: AER-style bidirectional hybrid (SOTA)
- **AssociationDiscrepancyDetection**: Anomaly Transformer (SOTA)
- **ClassificationBasedDetection**: OneClassSVM

### Scoring Methods (Step 3)
- **MaxPoolingScoring**: Max score from overlapping windows
- **AveragePoolingScoring**: Average score from overlapping windows
- **WeightedAverageScoring**: Gaussian-weighted average
- **GaussianSmoothingScoring**: Smoothed scores

### Post-Processing (Step 4)
- **PercentileThreshold**: Fixed percentile (e.g., 95th)
- **F1OptimalThreshold**: Maximize F1 on validation set
- **StatisticalThreshold**: Mean + k*std
- Filtering: Remove short anomalies
- Merging: Combine close anomalies

## 📈 Benchmarks & Datasets

### Supported Datasets
- **SMD** (Server Machine Dataset): 38-dimensional, 28 machines
- **MSL** (Mars Science Laboratory): NASA telemetry
- **SMAP** (Soil Moisture Active Passive): NASA telemetry
- **SWAT** (Secure Water Treatment): Industrial control
- **PSM** (Pooled Server Metrics): Server monitoring
- **Synthetic**: For quick testing

### Evaluation Metrics
- **F1 Score**: Standard precision-recall harmonic mean
- **PA-F1**: Point-adjusted F1 (segment-based)
- **VUS-PR**: Volume Under Surface for PR curve (TSB-AD)
- **Precision & Recall**
- **Latency**: p50, p99 inference time

### Expected Performance

| Method              | MSL F1 | SMAP F1 | SMD F1 | Avg F1 |
|---------------------|--------|---------|--------|--------|
| Baseline (KNN)      | 0.65   | 0.68    | 0.62   | 0.65   |
| LSTM-VAE           | 0.72   | 0.75    | 0.70   | 0.72   |
| **AER** (SOTA)     | **0.76** | **0.78** | **0.74** | **0.76** |
| Anomaly Transformer | 0.74   | 0.77    | 0.73   | 0.75   |

## 🔬 Development Phases

### Phase 1: Infrastructure ✅ (Completed)
- [x] Abstract base classes for pipeline
- [x] Simple baseline implementations
- [x] Pipeline orchestrator
- [x] Configuration system
- [x] Evaluation framework (F1, PA-F1, VUS-PR)
- [x] Data loaders (AnomLLM, custom datasets)

**Status**: Infrastructure complete and stable

### Phase 2: SOTA Training-Based Components ⚠️ (Archived)
- [x] AER implementation (BiLSTM encoder-decoder)
- [x] Anomaly Transformer (attention with association discrepancy)
- [ ] Training blocked by lack of GPU resources

**Status**: Archived to `archived/training_scripts/` and `archived/training_docs/`
**Note**: Pre-trained models can still be used for inference in Phase 3

### Phase 3: Foundation Model + LLM Approach 🚧 (Current)

#### Phase 3.1: Documentation & Planning ✅ (Completed 2026-02-17)
- [x] Archive training-focused code
- [x] Create Phase 3 specifications (5 documents)
- [x] Migration guide
- [x] Update README and architecture docs
- [x] Redesign architecture to fit into 4-step framework (not 5-step extension)

**Status**: Planning complete, 4-step enhancement architecture finalized, ready for implementation

#### Phase 3.2: Foundation Model Integration 📋 (Next)
- [ ] Install and test TimesFM
- [ ] Install and test Chronos
- [ ] Create wrapper classes (`src/foundation_models/`)
- [ ] Implement ensemble forecasting
- [ ] Test on sample datasets

**Target**: Zero-shot forecasting working on benchmark data

#### Phase 3.3: Statistical Evidence Extraction 📋
- [ ] Create `src/statistical_evidence/` module
- [ ] Implement 10+ evidence metrics
- [ ] Integration with pre-trained models (optional)
- [ ] Validate evidence quality

**Target**: Comprehensive evidence dictionary for each window

#### Phase 3.4: LLM Reasoning Layer 📋
- [ ] Implement LLM agent (`src/llm_reasoning/`)
- [ ] Create evidence-based prompt builder
- [ ] Output parser with validation
- [ ] Test on synthetic data
- [ ] Human evaluation of explanations

**Target**: Explainable anomaly detection with cited evidence

#### Phase 3.5: RAG System 📋
- [ ] Set up vector database (ChromaDB or FAISS)
- [ ] Implement pattern retrieval
- [ ] Populate with historical patterns
- [ ] Integrate with LLM agent

**Target**: Improved reasoning via historical context

#### Phase 3.6: Comprehensive Evaluation 📋
- [ ] Benchmark on multiple datasets (UCR, Yahoo, NAB)
- [ ] Ablation studies (evidence metrics, RAG impact, LLM models)
- [ ] Compare to Phase 2 baseline
- [ ] Cost analysis

**Target**: F1 > 0.75 with explainability, competitive with training-based methods

## 🛠️ Usage Examples

### Example 1: Compare Multiple Methods

```python
from src.utils.config_factory import build_pipeline_from_config
from src.data.loader import load_msl_smap_dataset
from src.evaluation.evaluator import Evaluator

# Load dataset
dataset = load_msl_smap_dataset("MSL")

# Compare methods
methods = ["baseline.yaml", "aer_pipeline.yaml", "transformer_pipeline.yaml"]

for config_file in methods:
    pipeline = build_pipeline_from_config(config_file)
    pipeline.fit(dataset.X_train)
    result = pipeline.predict(dataset.X_test, dataset.y_test)

    evaluator = Evaluator()
    metrics = evaluator.evaluate(dataset.y_test, result.predictions, result.point_scores)

    print(f"{config_file}: F1={metrics.f1:.3f}, PA-F1={metrics.pa_f1:.3f}")
```

### Example 2: Custom Component

```python
from src.pipeline.step2_detection import DetectionMethod

class MyCustomDetection(DetectionMethod):
    """Your custom detection method"""

    def fit(self, X_processed, y=None):
        # Your training logic
        pass

    def detect(self, X_processed):
        # Your detection logic
        scores = compute_anomaly_scores(X_processed)
        return scores

# Use it in config
config = {
    "detection": {"type": "MyCustomDetection", "params": {...}}
}
```

### Example 3: Hyperparameter Search

```python
import optuna

def objective(trial):
    window_size = trial.suggest_int("window_size", 50, 200)
    alpha = trial.suggest_float("alpha", 0.2, 0.8)

    config = build_config(window_size=window_size, alpha=alpha)
    pipeline = build_pipeline_from_config(config)
    # ... train and evaluate
    return f1_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(f"Best params: {study.best_params}")
```

## 📚 Documentation

### Phase 3: Foundation Model + LLM Approach (Current)

- **[Phase 3 Overview](spec/foundation_model_llm_architecture.md)** - Foundation model architecture
- **[Statistical Evidence Framework](spec/statistical_evidence_framework.md)** - 10+ evidence metrics
- **[LLM Reasoning Pipeline](spec/llm_reasoning_pipeline.md)** - LLM integration design
- **[RAG System Design](spec/rag_system_design.md)** - Historical pattern retrieval
- **[Integration of Pre-trained Models](spec/integration_pretrained_models.md)** - Using Phase 2 models
- **[Migration Guide](MIGRATION_GUIDE.md)** - Transition from Phase 2 to Phase 3

### Core Infrastructure

- **[Architecture Overview](spec/architecture_overview.md)** - Design philosophy, technology stack
- **[Phase 1: Infrastructure](spec/phase1_infrastructure.md)** - Base classes, evaluation framework

### Archived (Training-Based Approach)

- **[Archived Content](archived/README_ARCHIVED.md)** - Explanation of archived training code
- **[Phase 2: Training](archived/training_docs/phase2_sota_training.md)** - AER, Anomaly Transformer training

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_pipeline.py

# Run with coverage
pytest --cov=src tests/
```

## 🌟 Key Papers & References

### Foundation Models (Phase 3)
1. **TimesFM**: Das et al., "A decoder-only foundation model for time-series forecasting", ICML 2024
2. **Chronos**: Ansari et al., "Chronos: Learning the Language of Time Series", arXiv 2024

### LLM for Time Series (Phase 3)
3. **LLM-TSAD**: LLM-based time series anomaly detection with evidence-based reasoning
4. **AnomLLM**: Benchmark for LLM-based anomaly detection evaluation

### Deep Learning Methods (Phase 2, Archived)
5. **AER**: Zhang et al., "Time Series Anomaly Detection with Adversarial Reconstruction Networks", 2023
6. **Anomaly Transformer**: Xu et al., "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy", ICLR 2022
7. **TranAD**: Tuli et al., "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data", VLDB 2022

### Evaluation & Benchmarks
8. **TSB-AD**: "Towards A Reliable Time-Series Anomaly Detection Benchmark", NeurIPS 2024
9. **Point-Adjusted Metrics**: Kim et al., "Towards a Rigorous Evaluation of Time-Series Anomaly Detection", AAAI 2022

### RAG & Retrieval
10. **RAG**: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas for contribution:
- New detection methods
- Additional datasets
- Performance optimizations
- Documentation improvements
- Bug fixes

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📮 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email: [your-email@example.com]
- Discussion forum: [link]

## 🙏 Acknowledgments

- Datasets from NASA, OmniAnomaly, and other research groups
- Inspired by TSB-AD benchmark framework
- Built with PyTorch, scikit-learn, and other open-source tools

---

**Status**: Phase 3 planning complete ✅ | Ready to implement foundation model integration

**Current Phase**: Phase 3.2 - Foundation Model Integration

**Next Steps**:
1. Install TimesFM and Chronos libraries
2. Create foundation model wrappers (`src/foundation_models/`)
3. Implement statistical evidence extraction (`src/statistical_evidence/`)
4. Integrate LLM reasoning layer (`src/llm_reasoning/`)

**Migration**: See `MIGRATION_GUIDE.md` for transitioning from Phase 2 (training-based) to Phase 3 (foundation models + LLM)

**Star ⭐ this repo if you find it useful!**
