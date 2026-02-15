# Phase 1: Infrastructure (Week 1)

## Objectives

Build the foundational infrastructure that enables rapid experimentation with different anomaly detection approaches. This phase focuses on:

1. ✅ Creating abstract interfaces for all 4 pipeline steps
2. ✅ Implementing simple baseline components for each step
3. ✅ Building the pipeline orchestrator
4. ✅ Setting up configuration system
5. ✅ Implementing evaluation framework
6. ✅ Testing end-to-end with a simple baseline

**Goal**: By end of Phase 1, you should be able to run a complete anomaly detection experiment from config file to evaluation metrics.

---

## Deliverables

### 1. Abstract Base Classes

#### 1.1 Step 1: Data Processing Interface

**File**: `src/pipeline/step1_data_processing.py`

**Components to implement**:
```python
@dataclass
class WindowConfig:
    window_size: int = 100
    stride: int = 1
    padding: str = "same"

class DataProcessor(ABC):
    """Base class for Step 1: Data Processing"""

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on training data and transform"""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform test data using fitted parameters"""
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return dimension of processed output"""
        pass
```

**Simple implementation to build**:
- `RawWindowProcessor`: Just sliding windows + z-score normalization
  - Input: (T, D) time series
  - Create windows: (N, W, D)
  - Normalize each window with StandardScaler
  - Output: (N, W, D)

**Test criteria**:
- ✅ Window creation works with different strides
- ✅ Normalization preserves shape
- ✅ fit_transform and transform produce consistent results
- ✅ Handles edge cases (short time series, single dimension)

---

#### 1.2 Step 2: Detection Method Interface

**File**: `src/pipeline/step2_detection.py`

**Components to implement**:
```python
class DetectionMethod(ABC):
    """Base class for Step 2: Detection"""

    @abstractmethod
    def fit(self, X_processed: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit detection method on processed training data"""
        pass

    @abstractmethod
    def detect(self, X_processed: np.ndarray) -> np.ndarray:
        """Return sub-sequence level anomaly scores"""
        pass
```

**Simple implementation to build**:
- `DistanceBasedDetection`: K-nearest neighbors distance
  - Store training data
  - For each test window, compute distance to k-nearest training windows
  - Return average distance as anomaly score
  - Use sklearn's NearestNeighbors

**Test criteria**:
- ✅ Fit stores training data correctly
- ✅ Detect returns scores with correct shape (N,)
- ✅ Higher scores for outliers, lower for normal
- ✅ Works with different k values

---

#### 1.3 Step 3: Scoring Interface

**File**: `src/pipeline/step3_scoring.py`

**Components to implement**:
```python
class ScoringMethod(ABC):
    """Base class for Step 3: Convert sub-sequence scores to point-wise scores"""

    @abstractmethod
    def score(self,
              subsequence_scores: np.ndarray,
              window_size: int,
              stride: int,
              original_length: int) -> np.ndarray:
        """Convert sub-sequence scores to point-wise scores"""
        pass
```

**Simple implementations to build**:
1. `MaxPoolingScoring`: Each point gets max score from all windows containing it
2. `AveragePoolingScoring`: Each point gets average score from all windows containing it

**Test criteria**:
- ✅ Output has correct length (T,)
- ✅ Each point is covered by at least one window
- ✅ Edge points (start/end) handled correctly
- ✅ Works with different strides

---

#### 1.4 Step 4: Post-Processing Interface

**File**: `src/pipeline/step4_postprocessing.py`

**Components to implement**:
```python
class ThresholdDetermination(ABC):
    """Base class for threshold determination"""

    @abstractmethod
    def find_threshold(self,
                      scores: np.ndarray,
                      labels: Optional[np.ndarray] = None) -> float:
        """Determine threshold for anomaly detection"""
        pass

class PostProcessor:
    """Complete post-processing pipeline"""

    def process(self,
                scores: np.ndarray,
                labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Return binary predictions and threshold"""
        pass
```

**Simple implementations to build**:
1. `PercentileThreshold`: Use 95th percentile as threshold
2. `F1OptimalThreshold`: Search for threshold that maximizes F1 (needs validation labels)

**Additional methods in PostProcessor**:
- `_filter_short_anomalies()`: Remove anomaly segments shorter than min_length
- `_merge_close_anomalies()`: Merge anomalies with small gaps

**Test criteria**:
- ✅ Threshold is reasonable (not 0 or inf)
- ✅ Binary predictions have same length as scores
- ✅ F1-optimal finds better threshold than fixed percentile
- ✅ Filtering and merging work correctly

---

### 2. Pipeline Orchestrator

**File**: `src/pipeline/orchestrator.py`

**Component to implement**:
```python
@dataclass
class PipelineResult:
    predictions: np.ndarray       # Binary (T,)
    point_scores: np.ndarray      # Float (T,)
    subsequence_scores: np.ndarray  # Float (N,)
    threshold: float
    metadata: Dict[str, Any]
    execution_time: Dict[str, float]

class AnomalyDetectionPipeline:
    """Complete 4-step anomaly detection pipeline"""

    def __init__(self,
                 data_processor: DataProcessor,
                 detection_method: DetectionMethod,
                 scoring_method: ScoringMethod,
                 post_processor: PostProcessor):
        pass

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        """Train the pipeline"""
        pass

    def predict(self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None) -> PipelineResult:
        """Run full pipeline on test data"""
        pass
```

**Features**:
- Time each step independently
- Collect metadata from all components
- Handle errors gracefully
- Support both supervised (with labels) and unsupervised modes

**Test criteria**:
- ✅ End-to-end pipeline runs without errors
- ✅ Execution times are reasonable and recorded
- ✅ Output shapes are correct
- ✅ Can run multiple times with same results

---

### 3. Configuration System

**File**: `src/utils/config_factory.py`

**Functionality**:
```python
def build_pipeline_from_config(config: dict) -> AnomalyDetectionPipeline:
    """Factory function to build pipeline from config dict"""
    pass

def load_config(config_path: str) -> dict:
    """Load YAML config file"""
    pass
```

**Example config file**: `configs/pipelines/baseline.yaml`
```yaml
experiment:
  name: "baseline_knn"
  description: "Simple KNN-based baseline"

data_processing:
  type: "RawWindowProcessor"
  window_size: 100
  stride: 1

detection:
  type: "DistanceBasedDetection"
  params:
    method: "knn"
    k: 5

scoring:
  type: "MaxPoolingScoring"

postprocessing:
  threshold:
    type: "PercentileThreshold"
    params:
      percentile: 95.0
  min_anomaly_length: 1
  merge_gap: 0
```

**Test criteria**:
- ✅ Can load config from YAML file
- ✅ Factory creates correct component types
- ✅ Invalid configs raise clear errors
- ✅ Default values work when params omitted

---

### 4. Evaluation Framework

**File**: `src/evaluation/metrics.py`

**Metrics to implement**:

#### 4.1 Basic Metrics
```python
def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard F1 score"""
    pass

def compute_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Precision and recall"""
    pass
```

#### 4.2 Point-Adjusted Metrics
```python
def compute_point_adjusted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Point-adjusted F1 score

    For each anomaly segment in ground truth:
    - If ANY point in the segment is detected, count as TP
    - Otherwise, count as FN

    For each predicted anomaly segment:
    - If it overlaps with ANY ground truth segment, count as TP
    - Otherwise, count as FP
    """
    pass
```

**Reference**: "Towards a Rigorous Evaluation of Time-Series Anomaly Detection" (AAAI 2022)

#### 4.3 VUS-PR (Volume Under Surface)
```python
def compute_vus_pr(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Volume Under Surface for Precision-Recall curve

    More reliable metric than point-wise F1.
    Computes precision-recall curve with sliding threshold,
    then integrates the area.
    """
    pass
```

**Reference**: TSB-AD benchmark (NeurIPS 2024)

**Note**: For Phase 1, implement basic F1 and PA-F1. VUS-PR can be added in Phase 2 if needed.

---

**File**: `src/evaluation/evaluator.py`

**Component to implement**:
```python
@dataclass
class EvaluationResult:
    f1: float
    precision: float
    recall: float
    pa_f1: float
    vus_pr: Optional[float] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = None

class Evaluator:
    """Evaluate pipeline results"""

    def evaluate(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 scores: np.ndarray) -> EvaluationResult:
        """Compute all evaluation metrics"""
        pass

    def evaluate_pipeline(self,
                         pipeline: AnomalyDetectionPipeline,
                         X_test: np.ndarray,
                         y_test: np.ndarray) -> EvaluationResult:
        """Run pipeline and evaluate results"""
        pass
```

**Test criteria**:
- ✅ Metrics match sklearn for standard cases
- ✅ PA-F1 handles continuous anomaly segments correctly
- ✅ Edge cases don't crash (all zeros, all ones, etc.)
- ✅ Latency measurement is accurate

---

### 5. Data Loading

**File**: `src/data/loader.py`

**Functionality**:
```python
@dataclass
class Dataset:
    X_train: np.ndarray  # (T_train, D)
    y_train: np.ndarray  # (T_train,)
    X_val: np.ndarray    # (T_val, D)
    y_val: np.ndarray    # (T_val,)
    X_test: np.ndarray   # (T_test, D)
    y_test: np.ndarray   # (T_test,)
    name: str
    metadata: Dict[str, Any]

def load_dataset(dataset_name: str, data_dir: str = "data/") -> Dataset:
    """Load standard benchmark dataset"""
    pass

def create_synthetic_dataset(n_samples: int = 1000,
                            n_dims: int = 5,
                            anomaly_ratio: float = 0.05) -> Dataset:
    """Create synthetic dataset for testing"""
    pass
```

**Datasets to support**:
- Synthetic (for quick testing)
- SMD (Server Machine Dataset)
- MSL (Mars Science Laboratory)
- SMAP (Soil Moisture Active Passive)

**For Phase 1**: Focus on synthetic data. Real datasets can be added in Phase 2.

**Synthetic data generation**:
- Normal: Sine wave + Gaussian noise
- Anomalies: Random spikes, level shifts, or trend changes

**Test criteria**:
- ✅ Synthetic data has correct shape
- ✅ Anomaly ratio is approximately correct
- ✅ Train/val/test splits are proper
- ✅ Labels are binary (0=normal, 1=anomaly)

---

### 6. End-to-End Integration

**File**: `experiments/run_baseline.py`

**Script to implement**:
```python
#!/usr/bin/env python
"""Run baseline experiment"""

import yaml
from src.pipeline.orchestrator import AnomalyDetectionPipeline
from src.utils.config_factory import build_pipeline_from_config
from src.data.loader import create_synthetic_dataset
from src.evaluation.evaluator import Evaluator

def main():
    # Load config
    with open("configs/pipelines/baseline.yaml") as f:
        config = yaml.safe_load(f)

    # Create synthetic dataset
    dataset = create_synthetic_dataset(n_samples=2000, n_dims=5, anomaly_ratio=0.05)

    # Build pipeline
    pipeline = build_pipeline_from_config(config)

    # Train
    print("Training pipeline...")
    pipeline.fit(dataset.X_train, dataset.y_train)

    # Predict
    print("Running inference...")
    result = pipeline.predict(dataset.X_test, dataset.y_test)

    # Evaluate
    print("Evaluating...")
    evaluator = Evaluator()
    eval_result = evaluator.evaluate(
        y_true=dataset.y_test,
        y_pred=result.predictions,
        scores=result.point_scores
    )

    # Print results
    print(f"\n{'='*50}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"{'='*50}")
    print(f"F1 Score:          {eval_result.f1:.3f}")
    print(f"Precision:         {eval_result.precision:.3f}")
    print(f"Recall:            {eval_result.recall:.3f}")
    print(f"PA-F1 Score:       {eval_result.pa_f1:.3f}")
    print(f"Inference Time:    {eval_result.latency_ms:.2f} ms")
    print(f"{'='*50}")

    # Print pipeline timing breakdown
    print(f"\nPipeline Timing Breakdown:")
    for step, time in result.execution_time.items():
        print(f"  {step:20s}: {time:.4f}s")

if __name__ == "__main__":
    main()
```

**Success criteria**:
- ✅ Script runs end-to-end without errors
- ✅ Produces reasonable F1 score (> 0.5 on synthetic data)
- ✅ Timing information is printed
- ✅ Results are reproducible (same random seed → same results)

---

## Testing Strategy

### Unit Tests

**File**: `tests/test_data_processing.py`
- Test window creation with different parameters
- Test normalization
- Test edge cases (empty data, single point, etc.)

**File**: `tests/test_detection.py`
- Test KNN detection
- Test fit/predict workflow
- Test with different data shapes

**File**: `tests/test_scoring.py`
- Test max pooling
- Test average pooling
- Test score aggregation correctness

**File**: `tests/test_postprocessing.py`
- Test threshold determination
- Test filtering and merging
- Test edge cases

**File**: `tests/test_pipeline.py`
- Test end-to-end pipeline
- Test with different configurations
- Test reproducibility

**File**: `tests/test_evaluation.py`
- Test metric calculations
- Test against known ground truth
- Test edge cases (no anomalies, all anomalies, etc.)

### Integration Tests

**File**: `tests/test_integration.py`
- Load config → build pipeline → run experiment
- Test with synthetic data
- Verify output format
- Check reproducibility

---

## Development Timeline

### Day 1-2: Abstract Interfaces & Simple Implementations
- [ ] Create all 4 abstract base classes
- [ ] Implement RawWindowProcessor
- [ ] Implement DistanceBasedDetection (KNN)
- [ ] Implement MaxPooling and AveragePooling scoring
- [ ] Implement PercentileThreshold
- [ ] Write unit tests for each component

### Day 3: Pipeline Orchestrator
- [ ] Implement AnomalyDetectionPipeline class
- [ ] Add timing and metadata collection
- [ ] Write tests for pipeline
- [ ] Test end-to-end flow

### Day 4: Configuration System
- [ ] Implement config factory
- [ ] Create example YAML configs
- [ ] Add validation and error handling
- [ ] Test config loading

### Day 5: Evaluation Framework
- [ ] Implement basic metrics (F1, precision, recall)
- [ ] Implement point-adjusted F1
- [ ] Implement Evaluator class
- [ ] Write tests against known values

### Day 6: Data Loading & Synthetic Data
- [ ] Implement synthetic data generator
- [ ] Create Dataset dataclass
- [ ] Test data loading
- [ ] Verify data quality

### Day 7: Integration & Documentation
- [ ] Write run_baseline.py script
- [ ] Run end-to-end test
- [ ] Fix any issues
- [ ] Document all APIs
- [ ] Write README for Phase 1

---

## Success Criteria for Phase 1

At the end of Phase 1, you should be able to:

1. ✅ **Run a complete experiment from config file**
   - Load config → build pipeline → train → evaluate

2. ✅ **Get reasonable results on synthetic data**
   - F1 > 0.5 with KNN baseline
   - PA-F1 is computed correctly

3. ✅ **Swap components easily**
   - Change scoring method in config → works immediately
   - Change threshold method in config → works immediately

4. ✅ **Have well-tested codebase**
   - All core components have unit tests
   - Integration test passes
   - Code coverage > 70%

5. ✅ **Ready for Phase 2**
   - Clean APIs for adding new components
   - Infrastructure supports SOTA methods
   - Configuration system is flexible

---

## Deliverables Checklist

### Code
- [ ] `src/pipeline/step1_data_processing.py` with RawWindowProcessor
- [ ] `src/pipeline/step2_detection.py` with DistanceBasedDetection
- [ ] `src/pipeline/step3_scoring.py` with MaxPooling and AveragePooling
- [ ] `src/pipeline/step4_postprocessing.py` with thresholding
- [ ] `src/pipeline/orchestrator.py` with AnomalyDetectionPipeline
- [ ] `src/utils/config_factory.py` with factory functions
- [ ] `src/evaluation/metrics.py` with F1, PA-F1
- [ ] `src/evaluation/evaluator.py` with Evaluator
- [ ] `src/data/loader.py` with synthetic data generation
- [ ] `experiments/run_baseline.py` script

### Tests
- [ ] Unit tests for all 4 pipeline steps
- [ ] Unit tests for evaluation metrics
- [ ] Integration test for full pipeline
- [ ] All tests pass

### Configuration
- [ ] `configs/pipelines/baseline.yaml`
- [ ] Example configs with comments

### Documentation
- [ ] README.md for the project
- [ ] API documentation (docstrings)
- [ ] Instructions for running baseline
- [ ] Phase 1 completion report

---

## Dependencies

**File**: `requirements.txt`
```
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
torch>=2.0.0
pyyaml>=6.0
pytest>=7.2.0
pytest-cov>=4.0.0
```

**Optional** (for later phases):
```
wandb>=0.15.0
hydra-core>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Notes

- **Keep it simple**: Phase 1 is about infrastructure, not performance
- **Test thoroughly**: Good tests now save debugging time later
- **Document well**: Clear docstrings and examples
- **Think ahead**: Design APIs that will support SOTA methods in Phase 2

---

## Next Phase Preview

In Phase 2, we'll add:
- AER (BiLSTM encoder-decoder with hybrid loss)
- Anomaly Transformer (attention with association discrepancy)
- Advanced scoring methods
- Real datasets (SMD, MSL, SMAP)

The infrastructure from Phase 1 should make this straightforward - just implement new classes that inherit from the base classes!
