"""Quick test for Anomaly Transformer architecture and training"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.anomaly_transformer import (
    AnomalyTransformer,
    compute_association_discrepancy,
    compute_anomaly_score,
)
from src.pipeline.step1_data_processing import WindowConfig
from src.pipeline.step1_data_processing_sota import AnomalyTransformerProcessor
from src.pipeline.step2_detection_sota import AssociationDiscrepancyDetection


def test_model_forward():
    """Test Anomaly Transformer forward pass"""
    print("Testing Anomaly Transformer forward pass...")

    B, L, D = 8, 100, 1
    win_size = L

    model = AnomalyTransformer(
        win_size=win_size,
        input_dim=D,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    )

    x = torch.randn(B, L, D)
    output, series_list, prior_list = model(x)

    assert output.shape == (B, L, D), f"Output shape {output.shape} != {(B, L, D)}"
    assert len(series_list) == 2, f"Expected 2 series layers, got {len(series_list)}"
    assert len(prior_list) == 2, f"Expected 2 prior layers, got {len(prior_list)}"

    for i, (series, prior) in enumerate(zip(series_list, prior_list)):
        expected = (B, 4, L, L)  # 4 heads
        assert series.shape == expected, f"Layer {i} series shape {series.shape} != {expected}"
        assert prior.shape == expected, f"Layer {i} prior shape {prior.shape} != {expected}"

    print("[PASS] Forward pass OK")


def test_loss_computation():
    """Test minimax loss computation"""
    print("Testing loss computation...")

    B, L, D = 4, 50, 2

    model = AnomalyTransformer(
        win_size=L,
        input_dim=D,
        d_model=32,
        n_heads=2,
        n_layers=1,
    )

    x = torch.randn(B, L, D)
    output, series_list, prior_list = model(x)

    # Reconstruction loss
    rec_loss = torch.nn.functional.mse_loss(output, x)

    # Association discrepancy
    series_loss, prior_loss = compute_association_discrepancy(
        series_list, prior_list, L
    )

    assert rec_loss.dim() == 0, "Reconstruction loss should be scalar"
    assert series_loss.dim() == 0, "Series loss should be scalar"
    assert prior_loss.dim() == 0, "Prior loss should be scalar"

    # Test gradients
    loss = rec_loss + series_loss + prior_loss
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    print("[PASS] Loss computation OK")


def test_anomaly_score():
    """Test anomaly score computation"""
    print("Testing anomaly score computation...")

    B, L, D = 4, 50, 1

    model = AnomalyTransformer(
        win_size=L,
        input_dim=D,
        d_model=32,
        n_heads=2,
        n_layers=1,
    )

    x = torch.randn(B, L, D)

    with torch.no_grad():
        output, series_list, prior_list = model(x)
        scores = compute_anomaly_score(
            x, output, series_list, prior_list,
            win_size=L, temperature=50.0
        )

    assert scores.shape == (B, L), f"Scores shape {scores.shape} != {(B, L)}"
    assert not np.any(np.isnan(scores)), "NaN in scores"
    assert not np.any(np.isinf(scores)), "Inf in scores"
    assert np.all(scores >= 0), "Scores should be non-negative"

    print("[PASS] Anomaly score OK")


def test_processor_integration():
    """Test AnomalyTransformerProcessor integration"""
    print("Testing processor integration...")

    W, D = 50, 1
    N = 100
    windows = np.random.randn(N, W, D).astype(np.float32)

    processor = AnomalyTransformerProcessor(
        window_config=WindowConfig(window_size=W, stride=1),
        d_model=32,
        n_heads=2,
        n_layers=1,
        epochs=2,
        batch_size=16,
        device="cpu",
    )

    # fit_transform
    result = processor.fit_transform(windows)

    assert result.shape == (N, W), f"Expected ({N}, {W}), got {result.shape}"
    assert not np.any(np.isnan(result)), "NaN in output"
    assert not np.any(np.isinf(result)), "Inf in output"

    # transform
    test_windows = np.random.randn(20, W, D).astype(np.float32)
    result_test = processor.transform(test_windows)
    assert result_test.shape == (20, W), f"Transform shape {result_test.shape} != (20, {W})"

    print("[PASS] Processor integration OK")


def test_detection_method():
    """Test AssociationDiscrepancyDetection"""
    print("Testing detection method...")

    N, W = 100, 50
    scores_2d = np.random.rand(N, W)

    detector = AssociationDiscrepancyDetection()
    scores_1d = detector.detect(scores_2d)

    assert scores_1d.shape == (N,), f"Expected ({N},), got {scores_1d.shape}"
    assert not np.any(np.isnan(scores_1d)), "NaN in detection scores"

    print("[PASS] Detection method OK")


def test_end_to_end():
    """Test full end-to-end training"""
    print("Testing end-to-end training...")

    T, D = 200, 1
    W = 50

    # Generate data
    X_train = np.random.randn(T, D).astype(np.float32)
    X_test = np.random.randn(T, D).astype(np.float32)

    # Create windows manually
    def create_windows(X, window_size, stride=1):
        N = (len(X) - window_size) // stride + 1
        windows = np.array([X[i*stride:i*stride+window_size] for i in range(N)])
        return windows

    windows_train = create_windows(X_train, W)
    windows_test = create_windows(X_test, W)

    # Train processor
    processor = AnomalyTransformerProcessor(
        window_config=WindowConfig(window_size=W, stride=1),
        d_model=32,
        n_heads=2,
        n_layers=1,
        epochs=3,
        batch_size=16,
        device="cpu",
    )

    # Fit and transform
    train_scores = processor.fit_transform(windows_train)
    test_scores = processor.transform(windows_test)

    # Detect
    detector = AssociationDiscrepancyDetection()
    train_anomaly_scores = detector.detect(train_scores)
    test_anomaly_scores = detector.detect(test_scores)

    assert train_anomaly_scores.shape[0] == windows_train.shape[0]
    assert test_anomaly_scores.shape[0] == windows_test.shape[0]

    print("[PASS] End-to-end training OK")


if __name__ == "__main__":
    print("=" * 60)
    print("  Anomaly Transformer Architecture Tests")
    print("=" * 60)

    test_model_forward()
    test_loss_computation()
    test_anomaly_score()
    test_processor_integration()
    test_detection_method()
    test_end_to_end()

    print("\n" + "=" * 60)
    print("  All tests passed!")
    print("=" * 60)
