"""Pipeline Orchestrator - Tie all 4 steps together"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import time

from .step1_data_processing import DataProcessor
from .step2_detection import DetectionMethod
from .step3_scoring import ScoringMethod
from .step4_postprocessing import PostProcessor


@dataclass
class PipelineResult:
    """Complete pipeline output"""
    predictions: np.ndarray  # Binary predictions (T,)
    point_scores: np.ndarray  # Point-wise scores (T,)
    subsequence_scores: np.ndarray  # Sub-sequence scores (N,)
    threshold: float
    metadata: Dict[str, Any]
    execution_time: Dict[str, float]  # Time for each step


class AnomalyDetectionPipeline:
    """Complete 4-step anomaly detection pipeline

    Orchestrates:
    1. Data Processing (window + preprocessing)
    2. Detection (compute sub-sequence scores)
    3. Scoring (convert to point-wise scores)
    4. Post-processing (threshold + extract anomalies)
    """

    def __init__(self,
                 data_processor: DataProcessor,
                 detection_method: DetectionMethod,
                 scoring_method: ScoringMethod,
                 post_processor: PostProcessor):
        """Initialize pipeline with 4 components

        Args:
            data_processor: Step 1 - Data processing
            detection_method: Step 2 - Detection method
            scoring_method: Step 3 - Scoring method
            post_processor: Step 4 - Post-processing
        """
        self.data_processor = data_processor
        self.detection_method = detection_method
        self.scoring_method = scoring_method
        self.post_processor = post_processor
        self.execution_time = {}

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        """Train the pipeline on training data

        Args:
            X_train: Training time series (T, D)
            y_train: Optional training labels (T,) - usually all normal
        """
        print(f"Training pipeline on data shape: {X_train.shape}")

        # Step 1: Data processing
        print("  Step 1: Data processing...")
        t0 = time.time()
        X_processed = self.data_processor.process(X_train, fit=True)
        self.execution_time['step1_fit'] = time.time() - t0
        print(f"    Processed to {X_processed.shape} in {self.execution_time['step1_fit']:.2f}s")

        # Step 2: Fit detection method
        print("  Step 2: Fitting detection method...")
        t0 = time.time()
        self.detection_method.fit(X_processed, y_train)
        self.execution_time['step2_fit'] = time.time() - t0
        print(f"    Fitted in {self.execution_time['step2_fit']:.2f}s")

        print("Pipeline training complete!")

    def predict(self,
                X_test: np.ndarray,
                y_test: Optional[np.ndarray] = None) -> PipelineResult:
        """Run full pipeline on test data

        Args:
            X_test: Test time series (T, D)
            y_test: Optional test labels for threshold tuning (T,)

        Returns:
            PipelineResult with predictions, scores, and metadata
        """
        original_length = len(X_test)

        # Step 1: Data processing
        t0 = time.time()
        X_processed = self.data_processor.process(X_test, fit=False)
        t1 = time.time()

        # Pass forecast context from Step 1 to Step 2 (if applicable)
        if hasattr(self.detection_method, 'set_forecast_context'):
            forecasts = getattr(self.data_processor, 'get_forecasts', lambda: None)()
            train_stats = getattr(self.data_processor, 'get_train_statistics', lambda: None)()
            self.detection_method.set_forecast_context(
                forecast_results=forecasts or [],
                train_statistics=train_stats or {}
            )

        # Step 2: Detection (get sub-sequence scores)
        subsequence_scores = self.detection_method.detect(X_processed)
        t2 = time.time()

        # Pass evidence context from Step 2 to Step 3 (if applicable)
        if hasattr(self.scoring_method, 'set_evidence_context'):
            evidence = getattr(self.detection_method, 'get_evidence', lambda: None)()
            if evidence is not None:
                self.scoring_method.set_evidence_context(evidence, X_processed)

        # Step 3: Scoring (sub-sequence → point-wise)
        point_scores = self.scoring_method.score(
            subsequence_scores,
            window_size=self.data_processor.window_config.window_size,
            stride=self.data_processor.window_config.stride,
            original_length=original_length
        )
        t3 = time.time()

        # Step 4: Post-processing
        predictions, threshold = self.post_processor.process(point_scores, y_test)
        t4 = time.time()

        return PipelineResult(
            predictions=predictions,
            point_scores=point_scores,
            subsequence_scores=subsequence_scores,
            threshold=threshold,
            metadata=self._get_metadata(),
            execution_time={
                'step1_process': t1 - t0,
                'step2_detect': t2 - t1,
                'step3_score': t3 - t2,
                'step4_postprocess': t4 - t3,
                'total': t4 - t0
            }
        )

    def _get_metadata(self) -> Dict[str, Any]:
        """Collect metadata from all components"""
        metadata = {
            'data_processor': type(self.data_processor).__name__,
            'detection_method': type(self.detection_method).__name__,
            'scoring_method': type(self.scoring_method).__name__,
            'window_size': self.data_processor.window_config.window_size,
            'stride': self.data_processor.window_config.stride
        }
        # Include evidence info if available
        if hasattr(self.detection_method, 'get_evidence'):
            evidence = self.detection_method.get_evidence()
            if evidence is not None:
                metadata['evidence_count'] = len(evidence)
                metadata['has_evidence'] = True
        return metadata
