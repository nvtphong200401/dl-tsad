# Quick Start Guide

Get started with Best TSAD in 5 minutes!

## Installation

```bash
cd best-tsad
pip install -r requirements.txt
```

## Run Your First Experiment

```bash
python experiments/run_baseline.py
```

**Expected Output**:
```
============================================================
Best TSAD - Baseline Experiment
============================================================

Creating synthetic dataset...
  Train size: 1200 (all normal)
  Val size:   400 (10 anomalies)
  Test size:  400 (32 anomalies)

...

============================================================
RESULTS: baseline_knn
============================================================
F1 Score:          0.321
Precision:         0.218
Recall:            0.611
PA-F1 Score:       0.769
============================================================
```

## Run with Better Threshold

```bash
python experiments/run_experiment.py --config configs/pipelines/baseline_f1optimal.yaml
```

## Create Custom Pipeline

### Step 1: Create config file `my_pipeline.yaml`

```yaml
experiment:
  name: "my_custom_pipeline"

data_processing:
  type: "RawWindowProcessor"
  window_size: 50  # Smaller windows
  stride: 1

detection:
  type: "DistanceBasedDetection"
  params:
    k: 10  # More neighbors

scoring:
  type: "AveragePoolingScoring"  # Try average instead of max

postprocessing:
  threshold:
    type: "F1OptimalThreshold"
  min_anomaly_length: 5  # Longer anomalies
  merge_gap: 10
```

### Step 2: Run your config

```bash
python experiments/run_experiment.py --config my_pipeline.yaml
```

## Run Tests

```bash
pytest tests/test_pipeline.py -v
```

## Project Structure

```
best-tsad/
├── src/pipeline/          # 4-step pipeline components
├── src/evaluation/        # Metrics
├── src/data/              # Data loaders
├── configs/pipelines/     # Configuration files
├── experiments/           # Experiment runners
└── tests/                 # Unit tests
```

## Key Concepts

### 4-Step Pipeline

Every pipeline has 4 steps:

1. **Data Processing** - Window transformation + preprocessing
2. **Detection** - Compute anomaly scores for windows
3. **Scoring** - Convert window scores to point scores
4. **Post-Processing** - Apply threshold and extract anomalies

### Configuration

All experiments are driven by YAML configs. Change any component by editing the config file!

### Evaluation

Two key metrics:
- **F1 Score** - Standard point-wise metric
- **PA-F1** - Point-adjusted (segment-based) - more lenient for continuous anomalies

## Common Commands

```bash
# Run baseline
python experiments/run_baseline.py

# Run with custom config
python experiments/run_experiment.py --config path/to/config.yaml

# Run with more data
python experiments/run_experiment.py --n-samples 5000 --anomaly-ratio 0.02

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_pipeline.py::test_end_to_end_pipeline -v
```

## What's Next?

- See `README.md` for full documentation
- See `PHASE1_COMPLETE.md` for implementation details
- See `spec/` for detailed specifications
- Ready for Phase 2? Check `spec/phase2_sota_components.md`

## Need Help?

- Check the specs in `spec/`
- Look at example configs in `configs/pipelines/`
- Read the code - it's well-documented!

## Tips

1. **Start simple**: Run baseline first to understand the flow
2. **Use F1OptimalThreshold**: Much better than percentile if you have labels
3. **Tune window_size**: Bigger windows capture more context but slower
4. **Check PA-F1**: More important than F1 for continuous anomalies
5. **Visualize**: Add matplotlib to plot scores and predictions

Happy anomaly detecting! 🎉
