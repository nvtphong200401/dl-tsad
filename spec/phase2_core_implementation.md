# Phase 2: Core Implementation (Weeks 1-4)

## Overview

This phase implements the core components of the foundation model + LLM approach, enhancing each step of the 4-step pipeline with zero-shot forecasting, statistical evidence extraction, and LLM reasoning.

**Timeline**: 4 weeks
**Status**: 📋 Planning complete, ready to implement
**Prerequisites**: Phase 1 (Infrastructure) completed

---

## Objectives

1. ✅ Integrate foundation models (TimesFM, Chronos) into Step 1
2. ✅ Implement statistical evidence framework (10+ metrics) for Step 2
3. ✅ Build LLM reasoning layer for Step 3
4. ✅ Enhance post-processing for Step 4 to handle LLM outputs
5. ✅ Create end-to-end pipeline with all 4 enhanced steps
6. ✅ Test on sample datasets

**Goal**: By end of Phase 2, have a working foundation model + LLM pipeline that can perform zero-shot anomaly detection with explanations.

---

## Phase Breakdown

### Week 1: Foundation Model Integration (Step 1 Enhancement)

**Deliverable**: Working foundation model forecasting integrated into preprocessing

#### Tasks

**1.1 Environment Setup**
- [ ] Install foundation model libraries:
  ```bash
  pip install timesfm chronos-forecasting
  pip install torch transformers  # Dependencies
  ```
- [ ] Test TimesFM installation with simple example
- [ ] Test Chronos installation with simple example
- [ ] Verify models can run on CPU

**1.2 Create Foundation Model Wrappers**

**File**: `src/foundation_models/__init__.py`
```python
from .timesfm_wrapper import TimesFMWrapper
from .chronos_wrapper import ChronosWrapper
from .ensemble import EnsembleForecaster

__all__ = ['TimesFMWrapper', 'ChronosWrapper', 'EnsembleForecaster']
```

**File**: `src/foundation_models/timesfm_wrapper.py`
```python
import numpy as np
from typing import Dict, Optional
from timesfm import TimesFM

class TimesFMWrapper:
    """Wrapper for Google's TimesFM foundation model."""

    def __init__(self, model_name: str = "google/timesfm-1.0-200m"):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Load pre-trained TimesFM model."""
        self.model = TimesFM.from_pretrained(self.model_name)

    def forecast(
        self,
        train_data: np.ndarray,
        horizon: int,
        context_length: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate zero-shot forecasts.

        Args:
            train_data: Training window (T,) or (T, D)
            horizon: Number of steps to forecast
            context_length: Length of context to use

        Returns:
            dict with 'forecast' key containing predictions
        """
        if self.model is None:
            self.load_model()

        # TimesFM expects (batch, time) format
        if train_data.ndim == 1:
            train_data = train_data.reshape(1, -1)

        forecast = self.model.forecast(
            time_series=train_data,
            horizon=horizon,
            context_length=context_length or len(train_data)
        )

        return {
            'forecast': forecast.squeeze(),
            'model': 'timesfm'
        }
```

**File**: `src/foundation_models/chronos_wrapper.py`
```python
import numpy as np
from typing import Dict, Optional
from chronos import ChronosPipeline

class ChronosWrapper:
    """Wrapper for Amazon's Chronos foundation model."""

    def __init__(self, model_name: str = "amazon/chronos-t5-small"):
        self.model_name = model_name
        self.pipeline = None

    def load_model(self):
        """Load pre-trained Chronos model."""
        self.pipeline = ChronosPipeline.from_pretrained(self.model_name)

    def forecast(
        self,
        train_data: np.ndarray,
        horizon: int,
        num_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Generate probabilistic forecasts with quantiles.

        Args:
            train_data: Training window (T,) or (T, D)
            horizon: Number of steps to forecast
            num_samples: Number of samples for quantile estimation

        Returns:
            dict with 'forecast', 'quantiles', 'samples' keys
        """
        if self.pipeline is None:
            self.load_model()

        # Chronos expects (batch, time) format
        if train_data.ndim == 1:
            train_data = train_data.reshape(1, -1)

        # Generate samples
        samples = self.pipeline.predict(
            context=train_data,
            prediction_length=horizon,
            num_samples=num_samples
        )  # Shape: (batch, num_samples, horizon)

        samples = samples.squeeze(0)  # (num_samples, horizon)

        # Compute quantiles
        quantiles = {
            'P01': np.quantile(samples, 0.01, axis=0),
            'P10': np.quantile(samples, 0.10, axis=0),
            'P50': np.quantile(samples, 0.50, axis=0),
            'P90': np.quantile(samples, 0.90, axis=0),
            'P99': np.quantile(samples, 0.99, axis=0),
        }

        return {
            'forecast': quantiles['P50'],  # Median as point forecast
            'quantiles': quantiles,
            'samples': samples,
            'model': 'chronos'
        }
```

**File**: `src/foundation_models/ensemble.py`
```python
import numpy as np
from typing import Dict, List
from .timesfm_wrapper import TimesFMWrapper
from .chronos_wrapper import ChronosWrapper

class EnsembleForecaster:
    """Ensemble forecasting with TimesFM and Chronos."""

    def __init__(self, models: List[str] = ['timesfm', 'chronos']):
        self.models = {}
        if 'timesfm' in models:
            self.models['timesfm'] = TimesFMWrapper()
        if 'chronos' in models:
            self.models['chronos'] = ChronosWrapper()

    def forecast(
        self,
        train_data: np.ndarray,
        horizon: int,
        strategy: str = 'average'
    ) -> Dict:
        """
        Generate ensemble forecast.

        Args:
            train_data: Training window
            horizon: Forecast horizon
            strategy: 'average' or 'weighted'

        Returns:
            Ensemble forecast with uncertainty
        """
        forecasts = {}

        # Get forecasts from all models
        for name, model in self.models.items():
            forecasts[name] = model.forecast(train_data, horizon)

        # Ensemble point forecast
        if strategy == 'average':
            point_forecast = np.mean([
                f['forecast'] for f in forecasts.values()
            ], axis=0)
        else:
            # Weighted by model confidence (TODO: implement)
            point_forecast = forecasts['chronos']['forecast']

        # Get quantiles from Chronos (if available)
        quantiles = forecasts.get('chronos', {}).get('quantiles', None)

        return {
            'forecast': point_forecast,
            'quantiles': quantiles,
            'individual_forecasts': forecasts,
            'ensemble_strategy': strategy
        }
```

**1.3 Enhance Step 1 Data Processor**

**File**: `src/pipeline/step1_data_processing.py` (add new class)
```python
from src.foundation_models import EnsembleForecaster

class FoundationModelProcessor(DataProcessor):
    """Enhanced preprocessor with foundation model forecasting."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.window_size = config.get('window_size', 100)
        self.horizon = config.get('horizon', 100)
        self.forecaster = EnsembleForecaster()
        self.train_statistics = None

    def fit_transform(self, X: np.ndarray) -> Dict:
        """Fit on training data and generate forecasts."""
        # Create sliding windows
        windows = self._create_windows(X, self.window_size)

        # Normalize
        normalized_windows = self._normalize_windows(windows)

        # Compute training statistics
        self.train_statistics = {
            'mean': np.mean(X),
            'std': np.std(X),
            'quantiles': np.quantile(X, [0.01, 0.10, 0.50, 0.90, 0.99])
        }

        return {
            'windows': normalized_windows,
            'train_statistics': self.train_statistics,
            'mode': 'train'
        }

    def transform(self, X: np.ndarray) -> Dict:
        """Transform test data with foundation model forecasts."""
        # Create test windows
        windows = self._create_windows(X, self.window_size)
        normalized_windows = self._normalize_windows(windows)

        # Generate forecasts for each window
        forecasts = []
        for window in normalized_windows:
            forecast_result = self.forecaster.forecast(
                train_data=window,
                horizon=self.horizon
            )
            forecasts.append(forecast_result)

        return {
            'windows': normalized_windows,
            'forecasts': forecasts,  # List of forecast dicts
            'train_statistics': self.train_statistics,
            'mode': 'test'
        }
```

**1.4 Testing**

**File**: `tests/test_foundation_models.py`
```python
import numpy as np
import pytest
from src.foundation_models import TimesFMWrapper, ChronosWrapper, EnsembleForecaster

def test_timesfm_forecast():
    """Test TimesFM wrapper."""
    model = TimesFMWrapper()

    # Synthetic data
    train_data = np.sin(np.linspace(0, 4*np.pi, 100))

    result = model.forecast(train_data, horizon=20)

    assert 'forecast' in result
    assert len(result['forecast']) == 20

def test_chronos_forecast():
    """Test Chronos wrapper."""
    model = ChronosWrapper()

    train_data = np.sin(np.linspace(0, 4*np.pi, 100))

    result = model.forecast(train_data, horizon=20)

    assert 'forecast' in result
    assert 'quantiles' in result
    assert len(result['forecast']) == 20
    assert 'P50' in result['quantiles']

def test_ensemble_forecaster():
    """Test ensemble forecasting."""
    ensemble = EnsembleForecaster(['timesfm', 'chronos'])

    train_data = np.sin(np.linspace(0, 4*np.pi, 100))

    result = ensemble.forecast(train_data, horizon=20, strategy='average')

    assert 'forecast' in result
    assert 'individual_forecasts' in result
    assert len(result['forecast']) == 20
```

**Week 1 Success Criteria**:
- [ ] TimesFM can generate forecasts on sample data
- [ ] Chronos can generate probabilistic forecasts with quantiles
- [ ] Ensemble forecaster combines both models
- [ ] Step 1 processor enhanced with foundation model forecasting
- [ ] All tests pass

---

### Week 2: Statistical Evidence Extraction (Step 2 Enhancement)

**Deliverable**: Working evidence extractor that produces 10+ metrics per window

#### Tasks

**2.1 Create Evidence Extractor Module**

**File**: `src/evidence/__init__.py`
```python
from .evidence_extractor import StatisticalEvidenceExtractor
from .forecast_based import ForecastBasedEvidence
from .statistical_tests import StatisticalTestEvidence
from .distribution_based import DistributionBasedEvidence
from .pattern_based import PatternBasedEvidence

__all__ = [
    'StatisticalEvidenceExtractor',
    'ForecastBasedEvidence',
    'StatisticalTestEvidence',
    'DistributionBasedEvidence',
    'PatternBasedEvidence'
]
```

**File**: `src/evidence/evidence_extractor.py`
```python
import numpy as np
from typing import Dict, List, Optional
from .forecast_based import ForecastBasedEvidence
from .statistical_tests import StatisticalTestEvidence
from .distribution_based import DistributionBasedEvidence
from .pattern_based import PatternBasedEvidence

class StatisticalEvidenceExtractor:
    """Extract 10+ statistical evidence metrics for anomaly detection."""

    def __init__(self, config: Dict):
        self.config = config
        self.enabled_categories = config.get('enabled_categories', [
            'forecast_based',
            'statistical_tests',
            'distribution_based',
            'pattern_based'
        ])

        # Initialize evidence extractors
        self.extractors = {
            'forecast_based': ForecastBasedEvidence(),
            'statistical_tests': StatisticalTestEvidence(),
            'distribution_based': DistributionBasedEvidence(),
            'pattern_based': PatternBasedEvidence()
        }

    def extract(
        self,
        train_window: np.ndarray,
        test_window: np.ndarray,
        forecast_result: Dict,
        train_statistics: Dict
    ) -> Dict:
        """
        Extract comprehensive statistical evidence.

        Args:
            train_window: Historical training data
            test_window: Test window to analyze
            forecast_result: Output from foundation model
            train_statistics: Statistics from training data

        Returns:
            Evidence dictionary with 10+ metrics
        """
        evidence = {}

        # Category 1: Forecast-Based Evidence
        if 'forecast_based' in self.enabled_categories:
            forecast_evidence = self.extractors['forecast_based'].extract(
                actual=test_window,
                forecast=forecast_result['forecast'],
                quantiles=forecast_result.get('quantiles'),
                samples=forecast_result.get('samples')
            )
            evidence.update(forecast_evidence)

        # Category 2: Statistical Tests
        if 'statistical_tests' in self.enabled_categories:
            test_evidence = self.extractors['statistical_tests'].extract(
                test_window=test_window,
                train_mean=train_statistics['mean'],
                train_std=train_statistics['std']
            )
            evidence.update(test_evidence)

        # Category 3: Distribution-Based
        if 'distribution_based' in self.enabled_categories:
            dist_evidence = self.extractors['distribution_based'].extract(
                train_window=train_window,
                test_window=test_window
            )
            evidence.update(dist_evidence)

        # Category 4: Pattern-Based
        if 'pattern_based' in self.enabled_categories:
            pattern_evidence = self.extractors['pattern_based'].extract(
                train_window=train_window,
                test_window=test_window
            )
            evidence.update(pattern_evidence)

        return evidence
```

**2.2 Implement Evidence Extractors**

**File**: `src/evidence/forecast_based.py`
```python
import numpy as np
from scipy.stats import percentileofscore

class ForecastBasedEvidence:
    """Extract forecast error metrics."""

    def extract(self, actual, forecast, quantiles=None, samples=None):
        """Extract MAE, MSE, MAPE, quantile violations, surprise."""
        evidence = {}

        # 1. Mean Absolute Error
        mae = np.mean(np.abs(actual - forecast))
        evidence['mae'] = mae

        # 2. Mean Squared Error
        mse = np.mean((actual - forecast) ** 2)
        evidence['mse'] = mse

        # 3. MAPE (handle divide by zero)
        epsilon = 1e-10
        mape = np.mean(np.abs((actual - forecast) / (actual + epsilon))) * 100
        evidence['mape'] = mape

        # 4. Quantile Violations (if available)
        if quantiles is not None:
            violations = {
                'above_p99': np.any(actual > quantiles['P99']),
                'below_p01': np.any(actual < quantiles['P01']),
                'above_p90': np.any(actual > quantiles['P90']),
                'below_p10': np.any(actual < quantiles['P10'])
            }
            evidence['quantile_violations'] = violations
            evidence['any_extreme_violation'] = violations['above_p99'] or violations['below_p01']

        # 5. Surprise Score (if samples available)
        if samples is not None:
            from scipy.stats import gaussian_kde
            surprise_scores = []
            for i, val in enumerate(actual):
                kde = gaussian_kde(samples[:, i])
                likelihood = kde.pdf(val)
                surprise = -np.log(likelihood + 1e-10)
                surprise_scores.append(surprise)
            evidence['surprise_score'] = np.mean(surprise_scores)
            evidence['max_surprise'] = np.max(surprise_scores)

        return evidence
```

**File**: `src/evidence/statistical_tests.py` (implement Z-score, Grubbs, CUSUM)
**File**: `src/evidence/distribution_based.py` (implement KL divergence, Wasserstein)
**File**: `src/evidence/pattern_based.py` (implement ACF, volatility, trend)

See `spec/statistical_evidence_framework.md` for detailed implementations.

**2.3 Integrate Evidence Extraction into Step 2**

**File**: `src/pipeline/step2_detection.py` (add new class)
```python
from src.evidence import StatisticalEvidenceExtractor

class EvidenceBasedDetection(DetectionMethod):
    """Detection via statistical evidence extraction."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.evidence_extractor = StatisticalEvidenceExtractor(config)
        self.train_statistics = None

    def fit(self, X_processed: Dict, y=None):
        """Store training statistics."""
        self.train_statistics = X_processed.get('train_statistics')

    def detect(self, X_processed: Dict) -> List[Dict]:
        """Extract evidence for each test window."""
        windows = X_processed['windows']
        forecasts = X_processed['forecasts']

        evidence_list = []
        for i, (window, forecast) in enumerate(zip(windows, forecasts)):
            evidence = self.evidence_extractor.extract(
                train_window=None,  # TODO: Pass relevant train window
                test_window=window,
                forecast_result=forecast,
                train_statistics=self.train_statistics
            )
            evidence['window_index'] = i
            evidence_list.append(evidence)

        return evidence_list
```

**Week 2 Success Criteria**:
- [ ] All 10+ evidence metrics implemented and tested
- [ ] Evidence extractor integrated into Step 2
- [ ] Evidence extraction runs successfully on sample data
- [ ] Output format matches specification

---

### Week 3: LLM Reasoning Layer (Step 3 Enhancement)

**Deliverable**: Working LLM agent that analyzes evidence and produces explanations

#### Tasks

**3.1 Create LLM Integration Module**

**File**: `src/llm/__init__.py`
```python
from .llm_agent import LLMAnomalyAgent
from .backends import OpenAIBackend, GeminiBackend, ClaudeBackend
from .prompt_builder import PromptBuilder
from .output_parser import OutputParser

__all__ = [
    'LLMAnomalyAgent',
    'OpenAIBackend',
    'GeminiBackend',
    'ClaudeBackend',
    'PromptBuilder',
    'OutputParser'
]
```

**3.2 Implement LLM Backends**

See `spec/llm_reasoning_pipeline.md` for complete implementation of:
- `src/llm/backends.py` (OpenAI, Gemini, Claude wrappers)
- `src/llm/prompt_builder.py` (Evidence formatting, prompt templates)
- `src/llm/output_parser.py` (JSON parsing, validation)
- `src/llm/llm_agent.py` (Main LLM agent class)

**3.3 Integrate LLM into Step 3**

**File**: `src/pipeline/step3_scoring.py` (add new class)
```python
from src.llm import LLMAnomalyAgent, OpenAIBackend

class LLMReasoningScoring(ScoringMethod):
    """Scoring via LLM reasoning over evidence."""

    def __init__(self, config: Dict):
        super().__init__(config)

        # Initialize LLM backend
        backend_type = config.get('backend', 'openai')
        if backend_type == 'openai':
            backend = OpenAIBackend(
                api_key=os.getenv('OPENAI_API_KEY'),
                model=config.get('model', 'gpt-4-turbo')
            )
        # ... other backends

        self.llm_agent = LLMAnomalyAgent(backend)

    def score(self, evidence_list: List[Dict], time_series: np.ndarray) -> Dict:
        """
        Analyze evidence with LLM reasoning.

        Args:
            evidence_list: List of evidence dicts from Step 2
            time_series: Original time series for context

        Returns:
            LLM analysis with anomaly ranges, confidence, reasoning
        """
        results = []

        for i, evidence in enumerate(evidence_list):
            # Prepare time series context
            window_ts = {
                'timestamps': range(i, i + len(evidence)),  # TODO: actual timestamps
                'values': time_series[i:i+100]  # TODO: actual window values
            }

            # Analyze with LLM
            result = self.llm_agent.analyze_window(
                time_series=window_ts,
                evidence=evidence
            )
            results.append(result)

        return {
            'llm_results': results,
            'mode': 'llm_reasoning'
        }
```

**3.4 Testing**

**File**: `tests/test_llm_reasoning.py`
```python
import pytest
from src.llm import LLMAnomalyAgent, OpenAIBackend

@pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="No API key")
def test_llm_agent():
    """Test LLM agent with mock evidence."""
    backend = OpenAIBackend(os.getenv('OPENAI_API_KEY'))
    agent = LLMAnomalyAgent(backend)

    # Mock evidence
    evidence = {
        'mae': 2.5,
        'mae_anomalous': True,
        'z_score': 3.8,
        'volatility_ratio': 5.2
    }

    time_series = {
        'timestamps': list(range(100)),
        'values': np.random.randn(100)
    }

    result = agent.analyze_window(time_series, evidence)

    assert 'anomalies' in result
    assert 'overall_assessment' in result
```

**Week 3 Success Criteria**:
- [ ] LLM backends implemented for OpenAI, Gemini, Claude
- [ ] Prompt builder formats evidence correctly
- [ ] Output parser handles JSON responses
- [ ] LLM agent integrated into Step 3
- [ ] End-to-end test with real LLM API

---

### Week 4: Integration & Testing

**Deliverable**: Complete end-to-end pipeline with all 4 enhanced steps

#### Tasks

**4.1 Enhance Step 4 Post-Processing**

**File**: `src/pipeline/step4_postprocessing.py` (add methods)
```python
def parse_llm_output(llm_results: List[Dict], time_series_length: int) -> np.ndarray:
    """Convert LLM outputs to binary labels."""
    labels = np.zeros(time_series_length, dtype=int)

    for result in llm_results:
        for anomaly in result.get('anomalies', []):
            start = anomaly.get('start_index', 0)
            end = anomaly.get('end_index', time_series_length)
            if anomaly['confidence'] >= 0.5:
                labels[start:end+1] = 1

    return labels
```

**4.2 Update Pipeline Orchestrator**

**File**: `src/pipeline/orchestrator.py` (update)
```python
class AnomalyDetectionPipeline:
    """Enhanced pipeline orchestrator for Phase 2."""

    def predict(self, X_test, y_test=None):
        """Run full pipeline."""
        # Step 1: Data Processing + Foundation Forecasting
        X_processed = self.data_processor.transform(X_test)

        # Step 2: Detection via Evidence Extraction
        evidence_list = self.detection_method.detect(X_processed)

        # Step 3: Scoring via LLM Reasoning (or traditional)
        if isinstance(self.scoring_method, LLMReasoningScoring):
            scoring_result = self.scoring_method.score(evidence_list, X_test)
            # Parse LLM output
            predictions = parse_llm_output(
                scoring_result['llm_results'],
                len(X_test)
            )
        else:
            # Traditional scoring (max pooling, etc.)
            scores = self.scoring_method.score(evidence_list)
            # Step 4: Post-processing
            predictions = self.postprocessing.process(scores)

        return {
            'predictions': predictions,
            'evidence': evidence_list,
            'scoring_result': scoring_result if 'llm' in str(type(self.scoring_method)) else None
        }
```

**4.3 Create Configuration Examples**

**File**: `configs/pipelines/phase2_full_llm.yaml`
```yaml
experiment:
  name: "phase2_full_llm"
  description: "Complete foundation model + LLM pipeline"

step1_preprocessing:
  type: "FoundationModelProcessor"
  window_size: 100
  horizon: 100
  models: ['timesfm', 'chronos']
  ensemble_strategy: 'average'

step2_detection:
  type: "EvidenceBasedDetection"
  enabled_categories:
    - forecast_based
    - statistical_tests
    - distribution_based
    - pattern_based

step3_scoring:
  type: "LLMReasoningScoring"
  backend: "gemini"
  model: "gemini-1.5-pro"
  temperature: 0.0
  include_rag: false  # RAG added in Phase 3

step4_postprocessing:
  parse_llm: true
  extract_explanations: true

evaluation:
  metrics: ['f1', 'precision', 'recall', 'pa_f1', 'vus_pr']
```

**File**: `configs/pipelines/phase2_statistical_baseline.yaml`
```yaml
# Same as above but use traditional scoring instead of LLM
step3_scoring:
  type: "StatisticalAggregationScoring"
  aggregation: "weighted_average"
  weights:
    mae: 0.3
    z_score: 0.3
    volatility_ratio: 0.2
    kl_divergence: 0.2
```

**4.4 Integration Tests**

**File**: `tests/test_phase2_integration.py`
```python
import numpy as np
from src.utils.config_factory import build_pipeline_from_config

def test_full_pipeline_statistical():
    """Test pipeline with statistical baseline (no LLM)."""
    config = load_config('configs/pipelines/phase2_statistical_baseline.yaml')
    pipeline = build_pipeline_from_config(config)

    # Synthetic data
    train_data = np.sin(np.linspace(0, 10*np.pi, 1000))
    test_data = np.sin(np.linspace(0, 10*np.pi, 500))

    # Fit and predict
    pipeline.fit(train_data)
    result = pipeline.predict(test_data)

    assert 'predictions' in result
    assert 'evidence' in result
    assert len(result['predictions']) == len(test_data)

@pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="No API key")
def test_full_pipeline_llm():
    """Test pipeline with LLM reasoning."""
    config = load_config('configs/pipelines/phase2_full_llm.yaml')
    pipeline = build_pipeline_from_config(config)

    # Test on small sample
    train_data = np.sin(np.linspace(0, 4*np.pi, 200))
    test_data = np.sin(np.linspace(0, 2*np.pi, 100))

    pipeline.fit(train_data)
    result = pipeline.predict(test_data)

    assert 'predictions' in result
    assert 'evidence' in result
    assert 'scoring_result' in result
    assert result['scoring_result']['mode'] == 'llm_reasoning'
```

**4.5 Create Experiment Scripts**

**File**: `experiments/run_phase2_experiment.py`
```python
#!/usr/bin/env python
"""Run Phase 2 experiment on real dataset."""

import argparse
from src.utils.config_factory import build_pipeline_from_config
from src.data.loader import load_dataset
from src.evaluation.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', default='results/phase2_results.json')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    pipeline = build_pipeline_from_config(config)

    # Load dataset
    dataset = load_dataset(args.dataset)

    # Fit and predict
    print(f"Fitting pipeline on {args.dataset}...")
    pipeline.fit(dataset.X_train)

    print(f"Running prediction...")
    result = pipeline.predict(dataset.X_test, dataset.y_test)

    # Evaluate
    evaluator = Evaluator()
    metrics = evaluator.evaluate(
        dataset.y_test,
        result['predictions'],
        result.get('scores')
    )

    print("\n=== Results ===")
    print(f"F1 Score: {metrics.f1:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall: {metrics.recall:.3f}")
    print(f"PA-F1: {metrics.pa_f1:.3f}")

    # Save results
    save_results(args.output, metrics, result, config)

if __name__ == '__main__':
    main()
```

**Week 4 Success Criteria**:
- [ ] Complete end-to-end pipeline works on sample data
- [ ] Statistical baseline mode works (no LLM)
- [ ] LLM mode works with all backends (OpenAI, Gemini, Claude)
- [ ] Configuration system supports both modes
- [ ] Integration tests pass
- [ ] Experiment script runs on real dataset

---

## Phase 2 Completion Checklist

### Code Implementation
- [ ] Foundation model wrappers (TimesFM, Chronos, Ensemble)
- [ ] Statistical evidence extractors (10+ metrics)
- [ ] LLM backends (OpenAI, Gemini, Claude)
- [ ] Enhanced pipeline components (Steps 1-4)
- [ ] Pipeline orchestrator updated
- [ ] Configuration system for Phase 2

### Testing
- [ ] Unit tests for foundation models
- [ ] Unit tests for evidence extractors
- [ ] Unit tests for LLM components
- [ ] Integration tests for full pipeline
- [ ] Tests pass in both statistical and LLM modes

### Documentation
- [ ] API documentation for new modules
- [ ] Configuration examples
- [ ] Usage examples in notebooks
- [ ] Update README with Phase 2 status

### Validation
- [ ] Pipeline runs successfully on at least 2 benchmark datasets
- [ ] LLM provides coherent explanations
- [ ] Evidence metrics are reasonable
- [ ] Performance baseline established (F1 score recorded)

---

## Expected Outcomes

By end of Phase 2, you should have:

1. **Working Pipeline**: Complete 4-step pipeline with foundation models + LLM
2. **Two Operating Modes**:
   - Statistical baseline (no LLM, fast, free)
   - Full LLM reasoning (explainable, slower, API costs)
3. **Baseline Performance**: F1 > 0.70 on standard benchmarks
4. **Foundation for Phase 3**: Ready to add RAG system and optimizations

---

## Next Phase

**Phase 3: Advanced Features & Integration**
- RAG system for historical pattern retrieval
- Prompt optimization
- Cost optimization
- Multi-dataset evaluation

---

**Status**: Ready to implement
**Last Updated**: 2026-02-17
**Prerequisites**: Phase 1 complete, API keys configured
