# Archived Training-Based Source Code

This directory contains source code from the training-based Phase 2 approach, which was archived due to GPU constraints.

## Contents

### models/
- **aer.py** - AER (Auto-Encoder with Regression) BiLSTM implementation
- **anomaly_transformer.py** - Anomaly Transformer with association discrepancy
- **__init__.py** - Module initialization

These models contain full training implementations requiring GPU.

### pipeline/
- **step1_data_processing_sota.py** - SOTA data processors
  - `AERProcessor` - Trains AER model, extracts error features
  - `AnomalyTransformerProcessor` - Trains Anomaly Transformer
- **step2_detection_sota.py** - SOTA detection methods
  - `HybridDetection` - Combines AER reconstruction + regression errors
  - `AssociationDiscrepancyDetection` - Uses Anomaly Transformer attention

## Why Archived?

The training-based approach (Phase 2) was **archived on 2026-02-17** due to:
- Lack of GPU resources for training
- Pivot to foundation model + LLM approach (Phase 3)
- Focus on zero-shot, explainable anomaly detection

## Usage

These files can be referenced for:

1. **Understanding the training-based approach**
   - See how AER and Anomaly Transformer were implemented
   - Reference for research paper comparisons

2. **Creating inference-only wrappers (Phase 3)**
   - Can load pre-trained weights for inference
   - Use as optional evidence signals in Phase 3
   - See `spec/integration_pretrained_models.md`

3. **Resuming training (if GPU available)**
   - Restore files to active `src/` directory
   - Follow `archived/training_docs/phase2_sota_training.md`
   - Use `archived/training_scripts/` for training workflows

## Inference-Only Usage Example

If you have pre-trained weights and want to use for inference in Phase 3:

```python
import sys
sys.path.append('archived/src_training')

from models.aer import AERModel

# Load pre-trained model (inference only)
model = AERModel(input_dim=1, lstm_units=30)
model.load_state_dict(torch.load('pretrained/aer_weights.pth'))
model.eval()

# Run inference (no training)
with torch.no_grad():
    output = model(window_tensor)
```

Better approach: Create lightweight wrapper in `src/evidence/pretrained_models.py` (Phase 3).

## Related Documentation

- **Phase 2 Training Specs**: `archived/training_docs/phase2_sota_training.md`
- **Training Scripts**: `archived/training_scripts/`
- **Integration Guide**: `spec/integration_pretrained_models.md`
- **Migration Guide**: `MIGRATION_GUIDE.md`

## Status

- **Archived**: 2026-02-17
- **Reason**: Pivoted to foundation model + LLM approach
- **Current Phase**: Phase 2 (Core Implementation) - Foundation models + LLM
- **Future**: May be used for inference-only in Phase 3 (optional evidence)

---

**Note**: This code is kept for reference and potential future use. The active development follows the foundation model + LLM approach (Phase 3).
