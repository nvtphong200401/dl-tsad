#!/usr/bin/env python
"""Run inference using pre-trained models from Kaggle

After training models on Kaggle, download the .pth and .pkl files
to the models/ folder, then use this script for fast inference.
"""

import sys
import os
import argparse
import pickle
import torch
import json
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.aer import AERModel
from src.models.anomaly_transformer import AnomalyTransformer
from src.data.anomllm_loader import load_anomllm_category, get_all_categories
from src.pipeline import (
    WindowConfig, AnomalyDetectionPipeline,
    HybridDetection, AssociationDiscrepancyDetection,
    WeightedAverageScoring, MaxPoolingScoring,
    F1OptimalThreshold, PercentileThreshold, PostProcessor
)
from src.evaluation.evaluator import Evaluator


class PretrainedAERProcessor:
    """Data processor using pre-trained AER model"""

    def __init__(self, model_path: str, metadata_path: str, window_config: WindowConfig, device: str = 'cpu'):
        self.window_config = window_config
        self.device = device
        self.is_fitted = True

        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Recreate model (supports both old hidden_dim and new lstm_units metadata)
        lstm_units = self.metadata.get('lstm_units',
                         self.metadata.get('hidden_dim', 30))
        self.model = AERModel(
            input_dim=self.metadata['input_dim'],
            lstm_units=lstm_units,
            num_layers=self.metadata.get('num_layers', 1)
        ).to(device)

        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.input_dim = self.metadata['input_dim']

        print(f"Loaded pre-trained AER model (F1: {self.metadata.get('f1', 'N/A')})")

    def process(self, X, fit=False):
        """Process data (fit is ignored for pre-trained)"""
        windows = self._create_windows(X)
        return self._process_windows(windows)

    def _create_windows(self, X):
        """Create sliding windows"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, D = X.shape
        W = self.window_config.window_size
        S = self.window_config.stride
        windows = []
        for i in range(0, T - W + 1, S):
            windows.append(X[i:i + W])
        return np.array(windows)

    def _process_windows(self, windows):
        """Compute per-window error features [rec, reg_b, reg_f] matching AERProcessor"""
        with torch.no_grad():
            x_full = torch.FloatTensor(windows).to(self.device)  # (N, W, D)
            x_trimmed = x_full[:, 1:-1, :]                       # (N, W-2, D)

            ry, y, fy = self.model(x_trimmed)

            rec_error = torch.mean((x_trimmed - y) ** 2, dim=(1, 2))
            reg_error_b = torch.mean((x_full[:, 0] - ry) ** 2, dim=1)
            reg_error_f = torch.mean((x_full[:, -1] - fy) ** 2, dim=1)

            errors = torch.stack([rec_error, reg_error_b, reg_error_f], dim=1)

        return errors.cpu().numpy()


def run_pretrained_experiments(model_type: str = 'aer',
                               category: str = 'point',
                               models_dir: str = 'models'):
    """Run experiments using pre-trained models

    Args:
        model_type: 'aer' or 'transformer'
        category: AnomLLM category
        models_dir: Directory containing pre-trained models
    """

    print("="*70)
    print("Running with Pre-trained Models")
    print("="*70)
    print(f"Model: {model_type}")
    print(f"Category: {category}")

    # Load dataset
    print(f"\nLoading {category} dataset...")
    dataset = load_anomllm_category(category)
    print(f"Test size: {len(dataset.X_test)} ({dataset.y_test.sum()} anomalies)")

    # Load pre-trained model
    model_path = os.path.join(models_dir, f"{model_type}_model_{category}.pth")
    metadata_path = os.path.join(models_dir, f"{model_type}_model_{category}_metadata.pkl")

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print(f"\nPlease train the model on Kaggle and download it first.")
        print(f"See KAGGLE_GUIDE.md for instructions.")
        return

    # Build pipeline with pre-trained model
    print("\nBuilding pipeline with pre-trained model...")
    window_config = WindowConfig(window_size=100, stride=1)

    if model_type == 'aer':
        data_processor = PretrainedAERProcessor(model_path, metadata_path, window_config)
        detection_method = HybridDetection(comb="mult")
        scoring_method = WeightedAverageScoring()
    else:
        print("ERROR: Only AER is implemented for pre-trained loading")
        return

    post_processor = PostProcessor(
        threshold_method=F1OptimalThreshold(),
        min_anomaly_length=3,
        merge_gap=5
    )

    pipeline = AnomalyDetectionPipeline(
        data_processor=data_processor,
        detection_method=detection_method,
        scoring_method=scoring_method,
        post_processor=post_processor
    )

    # No training needed!
    print("Skipping training (using pre-trained model)...")

    # Predict on validation for threshold
    print("\nValidating for threshold tuning...")
    val_result = pipeline.predict(dataset.X_val, dataset.y_val)

    evaluator = Evaluator()
    val_eval = evaluator.evaluate(dataset.y_val, val_result.predictions, val_result.point_scores)
    print(f"Val F1: {val_eval.f1:.3f}, Threshold: {val_result.threshold:.4f}")

    # Predict on test
    print("\nTesting...")
    test_result = pipeline.predict(dataset.X_test, dataset.y_test)

    # Evaluate
    eval_result = evaluator.evaluate(dataset.y_test, test_result.predictions, test_result.point_scores)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"F1 Score:    {eval_result.f1:.3f}")
    print(f"Precision:   {eval_result.precision:.3f}")
    print(f"Recall:      {eval_result.recall:.3f}")
    print(f"PA-F1:       {eval_result.pa_f1:.3f}")
    print(f"Detected:    {test_result.predictions.sum()}/{dataset.y_test.sum()}")
    print(f"Inference:   {test_result.execution_time['total']:.4f}s")
    print("="*70)

    # Save results
    result = {
        'category': category,
        'model_type': model_type,
        'f1': eval_result.f1,
        'precision': eval_result.precision,
        'recall': eval_result.recall,
        'pa_f1': eval_result.pa_f1,
        'detected': int(test_result.predictions.sum()),
        'total_anomalies': int(dataset.y_test.sum()),
        'inference_time': test_result.execution_time['total']
    }

    # Save to results
    os.makedirs("src/results/synthetic", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"src/results/synthetic/{model_type}_{category}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {json_filename}")


def main():
    parser = argparse.ArgumentParser(description='Run with pre-trained models')
    parser.add_argument('--model', type=str, default='aer', choices=['aer', 'transformer'],
                      help='Model type')
    parser.add_argument('--category', type=str, default='point',
                      help='AnomLLM category')
    parser.add_argument('--models-dir', type=str, default='models',
                      help='Directory with pre-trained models')
    args = parser.parse_args()

    run_pretrained_experiments(args.model, args.category, args.models_dir)


if __name__ == "__main__":
    main()
