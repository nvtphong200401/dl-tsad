# Archived Training-Focused Components

## Overview

This directory contains code and documentation from **Phase 2: SOTA Components** that focused on training deep learning models for time series anomaly detection. These files were archived on **2026-02-17** as part of a strategic pivot to a **foundation model + LLM reasoning approach**.

## Why These Files Were Archived

**Problem:** The training-based approach required GPU resources for model training (Anomaly Transformer, AER BiLSTM), which were not available. This blocked research progress.

**Solution:** Pivot to a GPU-free approach using:
- Pre-trained foundation forecasting models (TimesFM, Chronos) for zero-shot predictions
- Statistical evidence extraction (10+ metrics) for anomaly signals
- LLM reasoning layer for contextual understanding and explainable outputs
- Optional use of pre-trained AER/Transformer models (inference only, no retraining)

## What's in This Archive

### `training_scripts/`
Training-focused experiment scripts that require GPU training:
- `test_aer.py` - AER BiLSTM training test script
- `kaggle_train.py` - Kaggle GPU training script for Anomaly Transformer

### `training_docs/`
Documentation focused on training approaches:
- `phase2_sota_training.md` - Original Phase 2 SOTA components specification
- `PHASE2_TRAINING_COMPLETE.md` - Phase 2 training implementation report

## What's Still Active

The following components remain in the active codebase because they support inference-only usage:

### Model Inference (No Training)
- `src/models/aer.py` - AER BiLSTM forward pass (inference only)
- `src/models/anomaly_transformer.py` - Anomaly Transformer forward pass (inference only)
- `experiments/run_with_pretrained.py` - Example of using pre-trained model weights

### Core Pipeline (Fully Functional)
- `src/pipeline/orchestrator.py` - Pipeline orchestration
- `src/pipeline/step1_data_processing.py` - Data preprocessing and windowing
- `src/pipeline/step3_scoring.py` - Scoring methods
- `src/pipeline/step4_postprocessing.py` - Post-processing and evaluation

### Evaluation Framework (Unchanged)
- `src/evaluation/metrics.py` - Evaluation metrics (F1, PA-F1, VUS-PR)
- `src/evaluation/evaluator.py` - Evaluation orchestration

### Data Loading (Unchanged)
- `src/data/loader.py` - Data loading utilities
- `src/data/anomllm_loader.py` - AnomLLM dataset loader

## How to Use Pre-trained Models

If you have pre-trained model weights from Phase 2, you can still use them for inference:

```python
import torch
from src.models.aer import AER
from src.models.anomaly_transformer import AnomalyTransformer

# Load pre-trained AER model
model = AER(window_size=100, input_dim=1)
model.load_state_dict(torch.load('path/to/aer_weights.pth'))
model.eval()

# Run inference (no training)
with torch.no_grad():
    anomaly_scores = model(input_windows)
```

See `experiments/run_with_pretrained.py` for a complete example.

## New Approach (Phase 3)

The new **foundation model + LLM reasoning approach** offers several advantages:

✅ **No GPU training required** - Use zero-shot foundation models
✅ **Faster experimentation** - No training loops or hyperparameter tuning
✅ **Explainable outputs** - LLM provides reasoning and cites evidence
✅ **Probabilistic predictions** - Foundation models provide uncertainty quantification
✅ **Multi-faceted detection** - 10+ independent statistical anomaly signals

See the new architecture documentation:
- `spec/foundation_model_llm_architecture.md` - New architecture overview
- `spec/statistical_evidence_framework.md` - Evidence extraction design
- `spec/llm_reasoning_pipeline.md` - LLM integration design
- `MIGRATION_GUIDE.md` - Transition guide from Phase 2 to Phase 3

## References

**LLM-TSAD Project:** The LLM reasoning layer is adapted from the LLM-TSAD project (located at `../LLM-TSAD/`), which demonstrated effective anomaly detection using GPT-4 with evidence-based prompting.

**AnomLLM Benchmark:** The AnomLLM project (located at `../AnomLLM/`) provides comprehensive benchmarks and evaluation protocols that are preserved in the new approach.

## Future Work

These training scripts may be revisited in the future if:
1. GPU resources become available
2. Hybrid approaches combining trained models with foundation models are explored
3. Fine-tuning foundation models on specific domains is needed

For now, the focus is on maximizing research progress with available resources using the foundation model + LLM reasoning approach.

---

**Questions?** See `MIGRATION_GUIDE.md` for detailed transition information or review the new architecture specifications in `spec/`.
