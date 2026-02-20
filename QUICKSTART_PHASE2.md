# Quick Start: Phase 2 Week 1

**Goal**: Verify that Phase 2 Week 1 (Foundation Models) is working

**Time**: 10-15 minutes

---

## Step 1: Install Dependencies (5 min)

```bash
cd best-tsad

# Install required libraries
pip install chronos-forecasting torch transformers numpy scipy scikit-learn

# Verify installation
python -c "import chronos; print('✓ Chronos installed')"
python -c "import torch; print('✓ PyTorch installed')"
```

---

## Step 2: Run Simple Foundation Model Test (2 min)

```bash
python experiments/test_foundation_models_simple.py
```

**Expected Output**:
```
Testing Chronos (Amazon)
Loaded Chronos model: amazon/chronos-t5-tiny
Generating forecast...
✓ Forecast generated successfully!

✓ PASS: chronos
Total: 1/3 tests passed
```

**If this works**: Foundation models are installed correctly! ✅

**If this fails**: Check error message and install missing dependencies

---

## Step 3: Run Pipeline Integration Test (5 min)

```bash
python experiments/test_phase2_pipeline_integration.py
```

**Expected Output**:
```
PHASE 2 PIPELINE INTEGRATION TEST

1. Creating synthetic data...
2. Creating pipeline components...
3. Creating pipeline...
4. Fitting pipeline on training data...
   Foundation model processor fitted on 901 windows
5. Checking forecast generation...
6. Running prediction on test data...
   Generating forecasts for 401 windows...
   Generated 401 forecasts
7. Verifying output format...
   ✓ Output format is Phase 1 compatible
   ✓ Forecast data available: 401 forecast results
8. Results:
   Predictions shape: (500,)
   Detected anomalies: X points
9. Execution time:
   step1_process: X.XXs
   step2_detect: X.XXs
   total: X.XXs

✓ TEST PASSED: Foundation models work with Phase 1 pipeline!
```

**If this works**: Phase 2 Week 1 is complete! ✅

---

## Step 4: Run pytest (Optional)

```bash
# Run all foundation model tests
pytest tests/test_foundation_models.py -v

# Or run as script
python tests/test_foundation_models.py
```

---

## What You Just Verified

### ✅ Foundation Models Working
- Chronos can generate probabilistic forecasts
- Forecasts include quantiles (P01-P99)
- Models load from HuggingFace

### ✅ Pipeline Integration Working
- FoundationModelProcessor extends Phase 1 DataProcessor
- Works with existing pipeline orchestrator
- Backward compatible (no breaking changes)
- Forecasts are stored for Step 2

### ✅ Ready for Week 2
- Foundation models generate forecasts ✅
- Forecasts available for evidence extraction ✅
- Can proceed to implement statistical evidence framework ✅

---

## Next Steps

### Option A: Proceed to Week 2 (Recommended)

**Implement Statistical Evidence Extraction**:
1. Read `spec/phase2_core_implementation.md` (Week 2 section)
2. Read `spec/statistical_evidence_framework.md` (detailed specs)
3. Create `src/evidence/evidence_extractor.py`
4. Implement 10+ evidence metrics

### Option B: Explore Current Implementation

```python
# Try the foundation models yourself
from src.foundation_models import ChronosWrapper
import numpy as np

# Create data
train = np.sin(np.linspace(0, 10*np.pi, 500))

# Generate forecast
model = ChronosWrapper(model_name="amazon/chronos-t5-tiny")
result = model.forecast(train, horizon=100, num_samples=50)

# Explore results
print(f"Forecast: {result['forecast']}")
print(f"P90 quantile: {result['quantiles']['P90']}")
print(f"Uncertainty: {result['uncertainty']}")
```

### Option C: Review Implementation

**Key files to review**:
- `src/foundation_models/chronos_wrapper.py` - Chronos integration
- `src/pipeline/step1_foundation_model_processor.py` - Pipeline integration
- `experiments/test_phase2_pipeline_integration.py` - How it all works together

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'chronos'"

**Solution**:
```bash
pip install chronos-forecasting
```

### "RuntimeError: Chronos forecast failed"

**Possible causes**:
1. Input data too short (need at least 50-100 points)
2. Input data has NaN/Inf values
3. Model download failed

**Solution**: Check input data or try smaller model:
```python
ChronosWrapper(model_name="amazon/chronos-t5-tiny")
```

### Test runs but takes a long time

**Normal**: First run downloads models (~500MB for Chronos tiny)
- Models are cached in `~/.cache/huggingface/`
- Subsequent runs will be faster

**Speed up**:
- Use `chronos-t5-tiny` (smallest, fastest)
- Reduce `num_samples` to 20-50
- Use fewer test windows

---

## Success Checklist

- [ ] `pip install chronos-forecasting` succeeds
- [ ] `test_foundation_models_simple.py` passes
- [ ] `test_phase2_pipeline_integration.py` passes
- [ ] Forecasts are generated for test windows
- [ ] Quantiles are available in forecast results

**If all checked**: ✅ Phase 2 Week 1 is complete!

---

## What's Next: Week 2

**Objective**: Statistical Evidence Extraction

**Reading**:
- `spec/phase2_core_implementation.md` (Week 2 section)
- `spec/statistical_evidence_framework.md`

**Implementation**:
- Create `src/evidence/` module
- Implement 10+ evidence metrics:
  - Forecast errors (MAE, MSE, quantile violations)
  - Statistical tests (Z-score, Grubbs, CUSUM)
  - Distribution metrics (KL divergence, Wasserstein)
  - Pattern analysis (ACF, volatility, trend)

**Timeline**: 1 week

---

**Status**: ✅ Phase 2 Week 1 Complete
**Next**: Phase 2 Week 2 - Statistical Evidence Extraction
**Questions**: See `spec/phase2_core_implementation.md`
