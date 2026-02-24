"""Configuration factory for building pipelines from YAML configs"""

import yaml
from typing import Dict, Any

from ..pipeline import (
    WindowConfig,
    RawWindowProcessor,
    DistanceBasedDetection,
    MaxPoolingScoring,
    AveragePoolingScoring,
    PercentileThreshold,
    F1OptimalThreshold,
    PostProcessor,
    AnomalyDetectionPipeline
)

# Phase 2 components
from ..pipeline.step1_foundation_model_processor import FoundationModelProcessor
from ..pipeline.step1_stl_processor import STLProcessor
from ..pipeline.step2_detection import EvidenceBasedDetection
from ..pipeline.step3_scoring import LLMReasoningScoring

# SOTA components (may not be available if archived)
try:
    from ..pipeline.step1_data_processing_sota import (
        AERProcessor,
        AnomalyTransformerProcessor
    )
    from ..pipeline.step2_detection_sota import (
        HybridDetection,
        AssociationDiscrepancyDetection
    )
    from ..pipeline.step3_scoring_sota import (
        WeightedAverageScoring,
        GaussianSmoothingScoring
    )
    _SOTA_AVAILABLE = True
except ImportError:
    _SOTA_AVAILABLE = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_pipeline_from_config(config: Dict[str, Any]) -> AnomalyDetectionPipeline:
    """Factory function to build pipeline from config dict"""
    data_processor = _build_data_processor(config['data_processing'])
    detection_method = _build_detection_method(config['detection'])
    scoring_method = _build_scoring_method(config['scoring'])
    post_processor = _build_post_processor(config['postprocessing'])

    return AnomalyDetectionPipeline(
        data_processor=data_processor,
        detection_method=detection_method,
        scoring_method=scoring_method,
        post_processor=post_processor
    )


def _build_data_processor(config: Dict[str, Any]):
    """Build data processor from config"""
    processor_type = config['type']
    window_size = config.get('window_size', 100)
    stride = config.get('stride', 1)
    params = config.get('params', {})

    window_config = WindowConfig(window_size=window_size, stride=stride)

    if processor_type == 'RawWindowProcessor':
        return RawWindowProcessor(window_config)
    elif processor_type == 'FoundationModelProcessor':
        return FoundationModelProcessor(
            window_config=window_config,
            forecast_horizon=params.get('forecast_horizon', 64),
            models=params.get('models', ['chronos']),
            chronos_model=params.get('chronos_model', 'amazon/chronos-t5-tiny'),
            num_samples=params.get('num_samples', 20),
            ensemble_strategy=params.get('ensemble_strategy', 'average'),
        )
    elif processor_type == 'STLProcessor':
        return STLProcessor(
            window_config=window_config,
            forecast_horizon=params.get('forecast_horizon'),
            period=params.get('period'),
            seasonal=params.get('seasonal', 7),
            trend=params.get('trend'),
            robust=params.get('robust', True),
            num_synthetic_samples=params.get('num_synthetic_samples', 50),
        )
    elif processor_type == 'AERProcessor' and _SOTA_AVAILABLE:
        return AERProcessor(window_config, **params)
    elif processor_type == 'AnomalyTransformerProcessor' and _SOTA_AVAILABLE:
        return AnomalyTransformerProcessor(window_config, **params)
    else:
        raise ValueError(f"Unknown data processor type: {processor_type}")


def _build_detection_method(config: Dict[str, Any]):
    """Build detection method from config"""
    detection_type = config['type']
    params = config.get('params', {})

    if detection_type == 'DistanceBasedDetection':
        k = params.get('k', 5)
        method = params.get('method', 'knn')
        return DistanceBasedDetection(k=k, method=method)
    elif detection_type == 'EvidenceBasedDetection':
        return EvidenceBasedDetection(
            enabled_categories=params.get('enabled_categories'),
            weights=params.get('weights'),
        )
    elif detection_type == 'HybridDetection' and _SOTA_AVAILABLE:
        return HybridDetection(**params)
    elif detection_type == 'AssociationDiscrepancyDetection' and _SOTA_AVAILABLE:
        return AssociationDiscrepancyDetection()
    else:
        raise ValueError(f"Unknown detection type: {detection_type}")


def _build_scoring_method(config: Dict[str, Any]):
    """Build scoring method from config"""
    scoring_type = config['type']
    params = config.get('params', {})

    if scoring_type == 'MaxPoolingScoring':
        return MaxPoolingScoring()
    elif scoring_type == 'AveragePoolingScoring':
        return AveragePoolingScoring()
    elif scoring_type == 'LLMReasoningScoring':
        return LLMReasoningScoring(
            backend_type=params.get('backend_type', 'azure_openai'),
            batch_size=params.get('batch_size', 10),
            temperature=params.get('temperature'),
            pre_filter_percentile=params.get('pre_filter_percentile', 80.0),
        )
    elif scoring_type == 'WeightedAverageScoring' and _SOTA_AVAILABLE:
        return WeightedAverageScoring()
    elif scoring_type == 'GaussianSmoothingScoring' and _SOTA_AVAILABLE:
        return GaussianSmoothingScoring(**params)
    else:
        raise ValueError(f"Unknown scoring type: {scoring_type}")


def _build_post_processor(config: Dict[str, Any]):
    """Build post-processor from config"""
    threshold_config = config['threshold']
    threshold_type = threshold_config['type']
    threshold_params = threshold_config.get('params', {})

    if threshold_type == 'PercentileThreshold':
        percentile = threshold_params.get('percentile', 95.0)
        threshold_method = PercentileThreshold(percentile=percentile)
    elif threshold_type == 'F1OptimalThreshold':
        threshold_method = F1OptimalThreshold()
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")

    min_anomaly_length = config.get('min_anomaly_length', 1)
    merge_gap = config.get('merge_gap', 0)

    return PostProcessor(
        threshold_method=threshold_method,
        min_anomaly_length=min_anomaly_length,
        merge_gap=merge_gap
    )
