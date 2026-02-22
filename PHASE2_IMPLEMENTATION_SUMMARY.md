# Phase 2 Implementation Summary

**Date**: 2026-02-21
**Status**: Weeks 1-3 complete, ready for Week 4

---

## Current Project Status

### Phase 1: Infrastructure (Complete)
```
src/
├── data/            Dataset loaders
├── evaluation/      Metrics (F1, PA-F1, VUS-PR)
├── pipeline/        Base classes, orchestrator
└── utils/           Configuration factory
```

### Phase 2: Core Implementation (Weeks 1-3 Complete)

#### Week 1: Foundation Model Integration (DONE)
```
src/foundation_models/
├── base.py                     Base class
├── timesfm_wrapper.py          TimesFM wrapper
├── chronos_wrapper.py          Chronos wrapper (GPU-enabled)
└── ensemble.py                 Ensemble forecaster

src/pipeline/
└── step1_foundation_model_processor.py  Enhanced Step 1
```

#### Week 2: Statistical Evidence Extraction (DONE)
```
src/evidence/
├── __init__.py                 Module exports
├── evidence_extractor.py       Main orchestrator (4 categories)
├── forecast_based.py           MAE, MSE, MAPE, quantile violations, surprise
├── statistical_tests.py        Z-score, Grubbs test, CUSUM
├── distribution_based.py       KL divergence, Wasserstein distance
└── pattern_based.py            ACF break, volatility spike, trend break

src/pipeline/
├── step2_detection.py          + EvidenceBasedDetection class
└── orchestrator.py             + forecast context passing (Step 1 → Step 2)
```

#### Week 3: LLM Reasoning Layer (DONE)
```
src/llm/
├── __init__.py                 Module exports
├── backends.py                 AzureOpenAI, Gemini, Claude backends
├── prompt_builder.py           Evidence formatting + system prompt
├── output_parser.py            JSON parsing with fallback handling
└── llm_agent.py                LLMAnomalyAgent (batch processing + retry)

src/pipeline/
├── step3_scoring.py            + LLMReasoningScoring class
└── orchestrator.py             + evidence context passing (Step 2 → Step 3)
```

#### Week 4: Integration & Testing (Next)
```
configs/             To be updated
experiments/         End-to-end evaluation on real datasets
```

---

## Pipeline Configurations

### Mode 1: Heuristic Scoring (no API key needed)
```
FoundationModelProcessor → EvidenceBasedDetection → MaxPooling → Threshold
        │                          │
        │ Chronos forecasts        │ 13 statistical metrics
        └──────────────────────────┘
              weighted sum aggregation
```

### Mode 2: LLM Scoring (requires API key)
```
FoundationModelProcessor → EvidenceBasedDetection → LLMReasoningScoring → Threshold
        │                          │                        │
        │ Chronos forecasts        │ 13 metrics             │ Azure GPT-4o / Gemini / Claude
        └──────────────────────────┘                        │
              evidence dicts → LLM prompt → confidence scores + reasoning
```

The LLM acts as an intelligent evidence aggregator: it reads all 13 metrics per window, reasons about convergent signals, and returns a confidence score (0-1) per window.

---

## How to Run

### Install Dependencies

```bash
cd best-tsad
conda activate anomllm

# Core dependencies
pip install chronos-forecasting torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install statsmodels scipy scikit-learn "numpy<2"

# LLM dependencies
pip install openai anthropic google-generativeai python-dotenv
```

### Configure API Keys

Edit `.env` in the project root:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

### Run Unit Tests (no GPU or API key needed)

```bash
# All tests (61 tests)
python -m pytest tests/ -v

# Evidence tests only
python -m pytest tests/test_evidence.py -v

# LLM tests only (uses mock backend)
python -m pytest tests/test_llm.py -v
```

### Run Pipeline Experiment

```bash
# Heuristic scoring (Chronos + evidence + max pooling)
python experiments/test_phase2_pipeline_integration.py

# LLM scoring (Chronos + evidence + Azure GPT-4o)
python experiments/test_phase2_pipeline_integration.py --llm

# LLM with specific backend
python experiments/test_phase2_pipeline_integration.py --llm --backend gemini
```

---

## Quick Start Examples

### Heuristic Pipeline (no API key)

```python
from src.pipeline.step1_foundation_model_processor import FoundationModelProcessor
from src.pipeline.step1_data_processing import WindowConfig
from src.pipeline.step2_detection import EvidenceBasedDetection
from src.pipeline.step3_scoring import MaxPoolingScoring
from src.pipeline.step4_postprocessing import PostProcessor, PercentileThreshold
from src.pipeline.orchestrator import AnomalyDetectionPipeline

pipeline = AnomalyDetectionPipeline(
    data_processor=FoundationModelProcessor(
        WindowConfig(window_size=100, stride=1),
        forecast_horizon=64,
        models=['chronos'],
        chronos_model="amazon/chronos-t5-tiny",
        num_samples=20
    ),
    detection_method=EvidenceBasedDetection(),
    scoring_method=MaxPoolingScoring(),
    post_processor=PostProcessor(PercentileThreshold(95.0))
)

pipeline.fit(X_train)
result = pipeline.predict(X_test, y_test)
```

### LLM Pipeline (requires .env with API key)

```python
from src.pipeline.step3_scoring import LLMReasoningScoring

pipeline = AnomalyDetectionPipeline(
    data_processor=FoundationModelProcessor(
        WindowConfig(window_size=100, stride=1),
        forecast_horizon=64,
        models=['chronos'],
        chronos_model="amazon/chronos-t5-tiny",
        num_samples=20
    ),
    detection_method=EvidenceBasedDetection(),
    scoring_method=LLMReasoningScoring(backend_type="azure_openai"),
    post_processor=PostProcessor(PercentileThreshold(95.0))
)

pipeline.fit(X_train)
result = pipeline.predict(X_test, y_test)

# Access LLM reasoning
llm_results = pipeline.scoring_method.get_llm_results()
print(f"LLM API calls: {pipeline.scoring_method.get_call_count()}")
```

---

## File Structure

```
best-tsad/
├── .env                                API keys (not committed)
├── .gitignore                          Ignores .env, __pycache__, etc.
├── src/
│   ├── foundation_models/              Week 1
│   │   ├── base.py
│   │   ├── timesfm_wrapper.py
│   │   ├── chronos_wrapper.py          GPU-enabled
│   │   └── ensemble.py
│   ├── evidence/                        Week 2
│   │   ├── __init__.py
│   │   ├── evidence_extractor.py       Orchestrates 4 categories
│   │   ├── forecast_based.py           5 metrics
│   │   ├── statistical_tests.py        3 metrics
│   │   ├── distribution_based.py       2 metrics
│   │   └── pattern_based.py            3 metrics
│   ├── llm/                             Week 3
│   │   ├── __init__.py
│   │   ├── backends.py                 Azure OpenAI, Gemini, Claude
│   │   ├── prompt_builder.py           Evidence formatting + prompts
│   │   ├── output_parser.py            JSON parsing with fallback
│   │   └── llm_agent.py               Batch processing + retry
│   ├── pipeline/
│   │   ├── step1_data_processing.py    Base + RawWindowProcessor
│   │   ├── step1_foundation_model_processor.py  Week 1
│   │   ├── step2_detection.py          + EvidenceBasedDetection (Week 2)
│   │   ├── step3_scoring.py            + LLMReasoningScoring (Week 3)
│   │   ├── step4_postprocessing.py     Percentile, F1Optimal thresholds
│   │   └── orchestrator.py             Context passing (Steps 1→2→3)
│   └── (rag/ ready for Phase 3)
│
├── tests/
│   ├── test_pipeline.py                Phase 1 pipeline tests (7 tests)
│   ├── test_foundation_models.py       Week 1 unit tests (5 tests)
│   ├── test_evidence.py                Week 2 unit tests (26 tests)
│   ├── test_evidence_integration.py    Week 2 integration tests (2 tests)
│   └── test_llm.py                     Week 3 unit tests (22 tests)
│
├── experiments/
│   ├── test_foundation_models_simple.py         Standalone Chronos test
│   └── test_phase2_pipeline_integration.py      Full pipeline (--llm flag)
│
└── archived/
    ├── src_training/                   Phase 1 training-based code
    └── training_scripts/              Archived tests
```

---

## Week 3 Technical Details

### LLM Backend Support

| Backend | Client | Env Vars | Model Default |
|---------|--------|----------|---------------|
| Azure OpenAI | `openai.AzureOpenAI` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` | gpt-4o |
| Gemini | `google.generativeai` | `GOOGLE_API_KEY` | gemini-2.0-flash |
| Claude | `anthropic.Anthropic` | `ANTHROPIC_API_KEY` | claude-3-5-sonnet-latest |

### How LLM Scoring Works

1. Orchestrator passes evidence dicts (from Step 2) to `LLMReasoningScoring`
2. Agent batches windows (default: 10 per LLM call) to reduce API costs
3. Each batch prompt contains: time series summary + 13 evidence metrics per window
4. LLM returns JSON with `is_anomaly`, `confidence`, `reasoning` per window
5. Confidence scores (0-1) replace the heuristic weighted sum
6. Max pooling converts window scores to point-wise scores
7. If LLM fails (API error, parse error), falls back to heuristic scores

### Context Injection Chain

```
Orchestrator.predict():
  Step 1: data_processor.process()
    ↓ hasattr check
  Step 1→2: detection_method.set_forecast_context(forecasts, train_stats)
  Step 2: detection_method.detect() → scores + evidence
    ↓ hasattr check
  Step 2→3: scoring_method.set_evidence_context(evidence, windows)
  Step 3: scoring_method.score() → point-wise scores
  Step 4: post_processor.process() → predictions
```

All context injection uses duck typing (`hasattr`). Existing Phase 1 components (DistanceBasedDetection, MaxPoolingScoring) work unchanged.

---

## Test Results

**61 passed, 1 skipped, 0 failures** (full test suite)

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_pipeline.py` | 7 | Phase 1 pipeline basics |
| `test_foundation_models.py` | 4+1 skip | Chronos, ensemble (TimesFM skipped) |
| `test_evidence.py` | 26 | All 4 evidence categories |
| `test_evidence_integration.py` | 2 | Pipeline with evidence detection |
| `test_llm.py` | 22 | Prompts, parsing, mock agent |

---

## Next Steps

### Week 4: Integration & Evaluation

1. Run full pipeline (heuristic + LLM) on real benchmark datasets
2. Compare: heuristic scoring vs LLM scoring vs Phase 1 baseline
3. Evaluate metrics: F1, Precision, Recall, PA-F1
4. Analyze LLM reasoning quality and cost

### Future: Phase 3 (RAG System)

- Vector database for historical pattern retrieval
- Inject similar past anomalies into LLM prompt
- Prompt optimization based on evaluation results

---

## Troubleshooting

### Issue: "chronos not found"
```bash
pip install chronos-forecasting
```

### Issue: NumPy incompatibility (_ARRAY_API not found)
```bash
pip install "numpy<2"
```

### Issue: PyTorch CPU-only (CUDA not available)
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: chronos-t5-tiny prediction degradation
```python
FoundationModelProcessor(..., forecast_horizon=64)  # Keep <= 64
```

### Issue: LLM API key not found
```bash
# Check .env file exists in project root with valid keys
cat .env
# Verify: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
```

### Issue: LLM JSON parsing fails
The output parser handles markdown code blocks, surrounding text, and partial JSON.
If the LLM returns non-JSON, it falls back to heuristic scoring automatically.

---

**Last Updated**: 2026-02-21
