# Architecture Overview - Best TSAD (Time Series Anomaly Detection)

## Project Goal
Build a state-of-the-art time series anomaly detection system using a modular 4-step pipeline that allows experimentation with different components at each stage to achieve the highest benchmark scores.

## Design Philosophy

### 1. Modularity
Each step in the pipeline is independent and replaceable. This allows:
- Testing different algorithms at each step
- Optimizing each component independently
- Easy comparison of different configurations
- Mix-and-match approach to find best combinations

### 2. Formal Pipeline Structure

Based on research literature, every anomaly detection method follows this structure:

```
Data Processing → Detection → Scoring → Post-Processing
    (Step 1)       (Step 2)    (Step 3)     (Step 4)
```

### Step 1: Data Processing
**Input:** Raw time series (T, D) - T timesteps, D dimensions
**Output:** Processed windows (N, W, D') - N windows, W window size, D' processed dimensions

**Components:**
- Window transformation (mandatory): Convert time series to sliding windows
- Pre-processing (optional): Feature extraction, neural embedding, model fitting

**Examples:**
- Raw windows + normalization
- Statistical feature extraction (mean, std, skewness, etc.)
- Neural embeddings (AutoEncoder, LSTM encoder)
- Model-specific processing (AER BiLSTM, Transformer embeddings)

### Step 2: Detection Method
**Input:** Processed windows (N, W, D')
**Output:** Sub-sequence anomaly scores (N,)

**Categories:**
- Distance-based: KNN, LOF, Isolation Forest
- Reconstruction-based: Compare original vs reconstructed
- Prediction-based: Compare actual vs predicted
- Hybrid: Combine multiple approaches (e.g., AER)
- Classification-based: OneClassSVM, hyperplane separation
- Association-based: Anomaly Transformer's association discrepancy

### Step 3: Scoring
**Input:** Sub-sequence scores (N,)
**Output:** Point-wise scores (T,)

**Purpose:** Convert window-level scores to point-level scores

**Methods:**
- Max pooling: Each point gets max score from overlapping windows
- Average pooling: Each point gets average score
- Weighted average: Gaussian or learned weights
- Smoothing: Gaussian filter, moving average

### Step 4: Post-Processing
**Input:** Point-wise scores (T,)
**Output:** Binary anomaly labels (T,)

**Components:**
- Threshold determination:
  - Percentile-based (e.g., 95th percentile)
  - F1-optimal (maximize F1 on validation set)
  - Statistical (mean + k*std)
  - Adaptive thresholding
- Anomaly extraction:
  - Point-wise vs interval detection
  - Filter short anomalies
  - Merge close anomalies

## Technology Stack

### Core Libraries
- **Python 3.9+**: Main language
- **NumPy**: Array operations
- **PyTorch**: Deep learning models (AER, Transformer)
- **scikit-learn**: Classical ML methods, metrics
- **scipy**: Statistical operations

### Configuration & Experiment Management
- **PyYAML**: Configuration files
- **hydra**: Advanced config management (optional)
- **wandb**: Experiment tracking (optional)

### Evaluation & Benchmarking
- **TSB-AD metrics**: VUS-PR implementation
- **Custom metrics**: Point-adjusted F1

## Project Structure

```
best-tsad/
├── spec/                          # Specifications (this folder)
│   ├── architecture_overview.md
│   ├── phase1_infrastructure.md
│   ├── phase2_sota_components.md
│   └── phase3_experiment_optimize.md
├── src/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── step1_data_processing.py    # Window + preprocessing
│   │   ├── step2_detection.py          # Detection methods
│   │   ├── step3_scoring.py            # Sub-sequence to point-wise
│   │   ├── step4_postprocessing.py     # Threshold + extraction
│   │   └── orchestrator.py             # Pipeline orchestrator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── aer.py                      # AER BiLSTM implementation
│   │   ├── anomaly_transformer.py      # Anomaly Transformer
│   │   └── tranad.py                   # TranAD
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                  # F1, PA-F1, VUS-PR, etc.
│   │   └── evaluator.py                # Evaluation orchestrator
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                   # Dataset loaders
│   │   └── normalization.py            # Normalization utilities
│   └── utils/
│       ├── __init__.py
│       └── config_factory.py           # Build pipeline from config
├── configs/
│   ├── datasets/
│   │   ├── smd.yaml
│   │   ├── msl.yaml
│   │   └── smap.yaml
│   ├── pipelines/
│   │   ├── aer_pipeline.yaml
│   │   ├── transformer_pipeline.yaml
│   │   └── baseline_pipeline.yaml
│   └── experiments/
│       └── compare_all.yaml
├── data/                           # Downloaded datasets
│   ├── SMD/
│   ├── MSL/
│   └── SMAP/
├── tests/
│   ├── test_pipeline.py
│   ├── test_data_processing.py
│   └── test_evaluation.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_results_analysis.ipynb
├── experiments/
│   └── runs/                       # Experiment results
├── requirements.txt
├── setup.py
└── README.md
```

## Key Design Patterns

### 1. Abstract Base Classes
Each step defines an abstract interface that all implementations must follow:
- `DataProcessor` (Step 1)
- `DetectionMethod` (Step 2)
- `ScoringMethod` (Step 3)
- `ThresholdDetermination` (Step 4)

### 2. Factory Pattern
Configuration files specify component types, and factories instantiate them:
```python
pipeline = build_pipeline_from_config(config)
```

### 3. Strategy Pattern
Each component can be swapped at runtime without changing the pipeline structure.

### 4. Pipeline Pattern
The orchestrator chains all 4 steps together and manages data flow.

## Evaluation Strategy

### Datasets
Start with standard benchmarks:
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

### Evaluation Protocol
1. Train on normal data (unsupervised)
2. Tune threshold on validation set
3. Report metrics on test set
4. Cross-validation for robust estimates
5. Statistical significance testing

## Success Criteria

### Performance Targets
- **Baseline**: F1 > 0.70 (Distance-based methods)
- **Target**: F1 > 0.75 (Match AER reported results)
- **Stretch**: F1 > 0.80 (Improved hybrid approach)

### Engineering Targets
- Modular design: Can swap any component in < 10 lines of code
- Fast experimentation: Run new config in < 5 minutes
- Reproducible: Same config produces same results
- Well-tested: > 80% code coverage

## LLM Agent Integration (Future)

After achieving SOTA detection, add LLM layer for:
1. Root cause analysis
2. Historical pattern matching (RAG)
3. Alert contextualization
4. False positive learning

**Note**: LLM is for explanation, not detection. Detection pipeline must work standalone.

## References

### Key Papers
1. AER: Auto-Encoder with Regression (Wong et al., 2022)
2. Anomaly Transformer (Xu et al., ICLR 2022)
3. TranAD (Tuli et al., VLDB 2022)
4. TSB-AD Benchmark (NeurIPS 2024)

### Datasets
- SMD: https://github.com/NetManAIOps/OmniAnomaly
- MSL/SMAP: https://github.com/khundman/telemanom
- SWAT: https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/

### Metrics Implementation
- TSB-AD: https://github.com/TheDatumOrg/TSB-AD
- Point-Adjusted Metrics: From "Towards a Rigorous Evaluation of Time-Series Anomaly Detection" (AAAI 2022)
