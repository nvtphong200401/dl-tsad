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

# SOTA components
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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file

    Args:
        config_path: Path to YAML config file

    Returns:
        Config dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_pipeline_from_config(config: Dict[str, Any]) -> AnomalyDetectionPipeline:
    """Factory function to build pipeline from config dict

    Args:
        config: Configuration dictionary

    Returns:
        Configured AnomalyDetectionPipeline
    """
    # Step 1: Build data processor
    data_processor = _build_data_processor(config['data_processing'])

    # Step 2: Build detection method
    detection_method = _build_detection_method(config['detection'])

    # Step 3: Build scoring method
    scoring_method = _build_scoring_method(config['scoring'])

    # Step 4: Build post-processor
    post_processor = _build_post_processor(config['postprocessing'])

    # Create pipeline
    pipeline = AnomalyDetectionPipeline(
        data_processor=data_processor,
        detection_method=detection_method,
        scoring_method=scoring_method,
        post_processor=post_processor
    )

    return pipeline


def _build_data_processor(config: Dict[str, Any]):
    """Build data processor from config"""
    processor_type = config['type']
    window_size = config.get('window_size', 100)
    stride = config.get('stride', 1)
    params = config.get('params', {})

    window_config = WindowConfig(window_size=window_size, stride=stride)

    if processor_type == 'RawWindowProcessor':
        return RawWindowProcessor(window_config)
    elif processor_type == 'AERProcessor':
        return AERProcessor(window_config, **params)
    elif processor_type == 'AnomalyTransformerProcessor':
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
    elif detection_type == 'HybridDetection':
        return HybridDetection(**params)
    elif detection_type == 'AssociationDiscrepancyDetection':
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
    elif scoring_type == 'WeightedAverageScoring':
        return WeightedAverageScoring()
    elif scoring_type == 'GaussianSmoothingScoring':
        return GaussianSmoothingScoring(**params)
    else:
        raise ValueError(f"Unknown scoring type: {scoring_type}")


def _build_post_processor(config: Dict[str, Any]):
    """Build post-processor from config"""
    threshold_config = config['threshold']
    threshold_type = threshold_config['type']
    threshold_params = threshold_config.get('params', {})

    # Build threshold method
    if threshold_type == 'PercentileThreshold':
        percentile = threshold_params.get('percentile', 95.0)
        threshold_method = PercentileThreshold(percentile=percentile)
    elif threshold_type == 'F1OptimalThreshold':
        threshold_method = F1OptimalThreshold()
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")

    # Build post-processor
    min_anomaly_length = config.get('min_anomaly_length', 1)
    merge_gap = config.get('merge_gap', 0)

    return PostProcessor(
        threshold_method=threshold_method,
        min_anomaly_length=min_anomaly_length,
        merge_gap=merge_gap
    )
