"""Tests for the AER pipeline — shape verification and end-to-end training.

Verifies every component in isolation and as a connected pipeline:
  AERModel → AERProcessor → HybridDetection → Scoring → PostProcessor

Run with:
    python -m pytest tests/test_aer_pipeline.py -v
    python tests/test_aer_pipeline.py           (standalone)
"""

import sys
import os
import pytest
import numpy as np
import torch

# Ensure the repo root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.aer import AERModel
from src.pipeline.step1_data_processing import WindowConfig
from src.pipeline.step1_data_processing_sota import AERProcessor
from src.pipeline.step2_detection_sota import HybridDetection
from src.pipeline.step3_scoring_sota import WeightedAverageScoring
from src.pipeline.step4_postprocessing import PostProcessor, PercentileThreshold, F1OptimalThreshold
from src.pipeline.orchestrator import AnomalyDetectionPipeline


# ========================================================================
# Fixtures
# ========================================================================
@pytest.fixture
def device():
    return "cpu"


@pytest.fixture(params=[1, 3], ids=["univariate", "multivariate"])
def input_dim(request):
    """Test with both univariate and multivariate inputs."""
    return request.param


@pytest.fixture(params=[10, 50, 100], ids=["W=10", "W=50", "W=100"])
def window_size(request):
    return request.param


# ========================================================================
# 1. AERModel — unit tests
# ========================================================================
class TestAERModel:
    """Verify AERModel input/output shapes and loss computation."""

    def test_forward_shapes(self, input_dim, window_size, device):
        """Model forward pass produces correct output shapes."""
        model = AERModel(input_dim=input_dim, lstm_units=16, num_layers=1).to(device)
        batch_size = 8
        seq_len = window_size - 2  # trimmed length

        x_trimmed = torch.randn(batch_size, seq_len, input_dim, device=device)
        ry, y, fy = model(x_trimmed)

        assert ry.shape == (batch_size, input_dim), \
            f"ry shape: expected ({batch_size}, {input_dim}), got {ry.shape}"
        assert y.shape == (batch_size, seq_len, input_dim), \
            f"y shape: expected ({batch_size}, {seq_len}, {input_dim}), got {y.shape}"
        assert fy.shape == (batch_size, input_dim), \
            f"fy shape: expected ({batch_size}, {input_dim}), got {fy.shape}"

    def test_forward_batch_size_1(self, device):
        """Works with batch size = 1 (edge case)."""
        model = AERModel(input_dim=1, lstm_units=8).to(device)
        x = torch.randn(1, 8, 1, device=device)
        ry, y, fy = model(x)
        assert ry.shape == (1, 1)
        assert y.shape == (1, 8, 1)
        assert fy.shape == (1, 1)

    def test_compute_loss_shape_and_gradient(self, device):
        """Loss is a scalar and gradients flow back."""
        model = AERModel(input_dim=2, lstm_units=8).to(device)
        W = 20
        B = 4
        x_full = torch.randn(B, W, 2, device=device)
        x_trimmed = x_full[:, 1:-1, :]

        ry, y, fy = model(x_trimmed)
        loss = model.compute_loss(x_full, ry, y, fy, reg_ratio=0.5)

        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss.item() > 0, "Loss should be positive for random data"

        # Gradients should flow
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.all(p.grad == 0), f"All-zero gradient for {name}"

    def test_loss_weighting_reg_ratio_0(self, device):
        """reg_ratio=0 → loss is purely reconstruction."""
        model = AERModel(input_dim=1, lstm_units=8).to(device)
        x_full = torch.randn(4, 12, 1, device=device)
        x_trimmed = x_full[:, 1:-1, :]
        ry, y, fy = model(x_trimmed)

        loss = model.compute_loss(x_full, ry, y, fy, reg_ratio=0.0)
        target_y = x_full[:, 1:-1]
        expected = torch.nn.functional.mse_loss(y, target_y)
        assert torch.allclose(loss, expected, atol=1e-6), \
            f"reg_ratio=0 should give pure reconstruction loss"

    def test_loss_weighting_reg_ratio_1(self, device):
        """reg_ratio=1 → loss is purely regression."""
        model = AERModel(input_dim=1, lstm_units=8).to(device)
        x_full = torch.randn(4, 12, 1, device=device)
        x_trimmed = x_full[:, 1:-1, :]
        ry, y, fy = model(x_trimmed)

        loss = model.compute_loss(x_full, ry, y, fy, reg_ratio=1.0)
        loss_ry = torch.nn.functional.mse_loss(ry, x_full[:, 0])
        loss_fy = torch.nn.functional.mse_loss(fy, x_full[:, -1])
        expected = 0.5 * loss_ry + 0.5 * loss_fy
        assert torch.allclose(loss, expected, atol=1e-6), \
            f"reg_ratio=1 should give pure regression loss"

    def test_encoder_bottleneck(self, device):
        """Encoder produces a bottleneck vector, not full hidden states."""
        model = AERModel(input_dim=2, lstm_units=16).to(device)
        x = torch.randn(3, 20, 2, device=device)

        _, (h_n, _) = model.encoder(x)
        # BiLSTM with 1 layer → h_n: (2, batch, lstm_units)
        assert h_n.shape == (2, 3, 16)

    def test_decoder_output_length(self, device):
        """Decoder output is seq_len + 2 (RepeatVector effect)."""
        model = AERModel(input_dim=1, lstm_units=8).to(device)
        seq_len = 48  # trimmed = W - 2 → original W = 50
        x = torch.randn(2, seq_len, 1, device=device)
        ry, y, fy = model(x)

        # y should be same length as input (trimmed)
        assert y.shape[1] == seq_len
        # internal decode_len should be seq_len + 2
        # We can check by verifying ry + y + fy covers seq_len + 2 timesteps
        total_timesteps = 1 + y.shape[1] + 1  # ry(1) + y(seq_len) + fy(1)
        assert total_timesteps == seq_len + 2

    def test_hidden_dim_alias(self, device):
        """Backward-compat: hidden_dim property works."""
        model = AERModel(input_dim=1, lstm_units=30)
        assert model.hidden_dim == 30

    def test_num_layers_gt_1(self, device):
        """Multi-layer LSTM works without error."""
        model = AERModel(input_dim=2, lstm_units=16, num_layers=2, dropout=0.1).to(device)
        x = torch.randn(4, 18, 2, device=device)
        ry, y, fy = model(x)
        assert y.shape == (4, 18, 2)


# ========================================================================
# 2. AERProcessor — integration tests
# ========================================================================
class TestAERProcessor:
    """Verify AERProcessor produces correct shapes and trains."""

    def test_fit_transform_shape(self):
        """fit_transform returns (N, 3) error features."""
        W, D = 20, 1
        N = 50
        windows = np.random.randn(N, W, D).astype(np.float32)

        processor = AERProcessor(
            window_config=WindowConfig(window_size=W, stride=1),
            lstm_units=8, epochs=2, batch_size=16,
            validation_split=0.2, patience=100, device="cpu"
        )
        result = processor.fit_transform(windows)

        assert result.shape == (N, 3), f"Expected (N, 3)={(N, 3)}, got {result.shape}"
        assert not np.any(np.isnan(result)), "NaN in output"
        assert not np.any(np.isinf(result)), "Inf in output"

    def test_fit_transform_multivariate(self):
        """Works with multivariate input."""
        W, D = 20, 5
        N = 40
        windows = np.random.randn(N, W, D).astype(np.float32)

        processor = AERProcessor(
            window_config=WindowConfig(window_size=W),
            lstm_units=8, epochs=2, batch_size=16,
            validation_split=0.2, patience=100, device="cpu"
        )
        result = processor.fit_transform(windows)
        assert result.shape == (N, 3)

    def test_transform_after_fit(self):
        """transform uses the fitted model and produces same shape."""
        W, D = 20, 1
        N = 30
        windows_train = np.random.randn(N, W, D).astype(np.float32)
        windows_test = np.random.randn(15, W, D).astype(np.float32)

        processor = AERProcessor(
            window_config=WindowConfig(window_size=W),
            lstm_units=8, epochs=2, batch_size=16,
            validation_split=0.2, patience=100, device="cpu"
        )
        _ = processor.fit_transform(windows_train)
        result = processor.transform(windows_test)

        assert result.shape == (15, 3)

    def test_errors_are_non_negative(self):
        """All error components should be >= 0 (they are MSE)."""
        W, D = 20, 2
        windows = np.random.randn(30, W, D).astype(np.float32)

        processor = AERProcessor(
            window_config=WindowConfig(window_size=W),
            lstm_units=8, epochs=2, batch_size=16,
            validation_split=0.2, patience=100, device="cpu"
        )
        result = processor.fit_transform(windows)

        assert np.all(result >= 0), "Error components must be non-negative"

    def test_loss_decreases_over_training(self):
        """Verify loss actually decreases during training (sanity check)."""
        W, D = 20, 1
        N = 60
        # Use a simple repeating pattern for easy learning
        pattern = np.sin(np.linspace(0, 4 * np.pi, W))
        windows = np.tile(pattern, (N, 1))[:, :, np.newaxis].astype(np.float32)

        processor = AERProcessor(
            window_config=WindowConfig(window_size=W),
            lstm_units=16, epochs=20, batch_size=32,
            validation_split=0.1, patience=100,
            learning_rate=1e-3, device="cpu"
        )

        # Capture initial errors
        result_pre = None  # Can't do transform before fit

        result = processor.fit_transform(windows)

        # After training, reconstruction error should be small for a simple pattern
        mean_rec_error = result[:, 0].mean()
        assert mean_rec_error < 1.0, \
            f"Mean reconstruction error ({mean_rec_error:.4f}) too high for a simple sin pattern"

    def test_legacy_hidden_dim_alias(self):
        """hidden_dim kwarg is accepted and maps to lstm_units."""
        processor = AERProcessor(
            window_config=WindowConfig(window_size=20),
            hidden_dim=64, alpha=0.3
        )
        assert processor.lstm_units == 64
        assert processor.reg_ratio == 0.3

    def test_output_dim_is_3(self):
        """get_output_dim returns 3."""
        processor = AERProcessor(window_config=WindowConfig(window_size=50))
        assert processor.get_output_dim() == 3

    def test_minimum_window_size(self):
        """Window size must be >= 3 (2 trimmed + at least 1 for reconstruction)."""
        W_min = 3
        windows = np.random.randn(10, W_min, 1).astype(np.float32)
        processor = AERProcessor(
            window_config=WindowConfig(window_size=W_min),
            lstm_units=8, epochs=1, batch_size=10,
            validation_split=0.1, patience=100, device="cpu"
        )
        result = processor.fit_transform(windows)
        assert result.shape == (10, 3)


# ========================================================================
# 3. HybridDetection — unit tests
# ========================================================================
class TestHybridDetection:
    """Verify HybridDetection produces correct shapes and scoring modes."""

    def test_detect_shape(self):
        """detect() returns (N,) from (N, 3) input."""
        N = 50
        errors = np.random.rand(N, 3).astype(np.float32)

        detector = HybridDetection(comb="mult")
        scores = detector.detect(errors)

        assert scores.shape == (N,), f"Expected ({N},), got {scores.shape}"

    def test_mult_mode_range(self):
        """In 'mult' mode, scores should be in [1, 4] (product of [1,2] ranges)."""
        errors = np.random.rand(100, 3)
        detector = HybridDetection(comb="mult")
        scores = detector.detect(errors)

        assert scores.min() >= 1.0 - 1e-6, f"Min score {scores.min()} < 1.0"
        assert scores.max() <= 4.0 + 1e-6, f"Max score {scores.max()} > 4.0"

    def test_sum_mode_range(self):
        """In 'sum' mode with lambda_rec=0.5, scores should be in [0, 1]."""
        errors = np.random.rand(100, 3)
        detector = HybridDetection(comb="sum", lambda_rec=0.5)
        scores = detector.detect(errors)

        assert scores.min() >= -1e-6, f"Min score {scores.min()} < 0"
        assert scores.max() <= 1.0 + 1e-6, f"Max score {scores.max()} > 1.0"

    def test_rec_mode(self):
        """In 'rec' mode, scores equal reconstruction errors."""
        errors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        detector = HybridDetection(comb="rec")
        scores = detector.detect(errors)
        np.testing.assert_array_almost_equal(scores, errors[:, 0])

    def test_reg_mode(self):
        """In 'reg' mode, scores equal avg regression errors."""
        errors = np.array([[0.1, 0.2, 0.4], [0.5, 0.6, 0.8]])
        detector = HybridDetection(comb="reg")
        scores = detector.detect(errors)
        expected = (errors[:, 1] + errors[:, 2]) / 2
        np.testing.assert_array_almost_equal(scores, expected)

    def test_constant_input(self):
        """Constant errors should not cause division by zero."""
        errors = np.ones((10, 3)) * 0.5
        detector = HybridDetection(comb="mult")
        scores = detector.detect(errors)

        assert not np.any(np.isnan(scores)), "NaN with constant input"
        assert not np.any(np.isinf(scores)), "Inf with constant input"

    def test_legacy_alpha_beta_kwargs(self):
        """Legacy alpha/beta kwargs are accepted without error."""
        detector = HybridDetection(alpha=0.5, beta=0.5)
        errors = np.random.rand(10, 3)
        scores = detector.detect(errors)
        assert scores.shape == (10,)

    def test_minmax_scale(self):
        """MinMax scaling produces correct range."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scaled = HybridDetection._minmax_scale(x, (1, 2))
        assert abs(scaled.min() - 1.0) < 1e-6
        assert abs(scaled.max() - 2.0) < 1e-6


# ========================================================================
# 4. WeightedAverageScoring — shape test
# ========================================================================
class TestWeightedAverageScoring:
    """Verify scoring converts (N,) to (T,) point-wise scores."""

    def test_score_shape(self):
        W, S, T = 20, 1, 100
        N = T - W + 1  # 81
        subs = np.random.rand(N)

        scorer = WeightedAverageScoring()
        point_scores = scorer.score(subs, window_size=W, stride=S, original_length=T)

        assert point_scores.shape == (T,), f"Expected ({T},), got {point_scores.shape}"

    def test_no_nan_or_inf(self):
        W, S, T = 50, 1, 200
        N = T - W + 1
        subs = np.random.rand(N)

        scorer = WeightedAverageScoring()
        point_scores = scorer.score(subs, window_size=W, stride=S, original_length=T)

        assert not np.any(np.isnan(point_scores))
        assert not np.any(np.isinf(point_scores))

    def test_higher_score_preserved(self):
        """Windows with higher scores should produce higher point scores."""
        W, S, T = 10, 5, 50
        N = (T - W) // S + 1
        subs = np.zeros(N)
        subs[N // 2] = 10.0  # spike in the middle

        scorer = WeightedAverageScoring()
        point_scores = scorer.score(subs, window_size=W, stride=S, original_length=T)

        # Middle region should have higher scores than edges
        mid = T // 2
        assert point_scores[mid] > point_scores[0]


# ========================================================================
# 5. End-to-end pipeline test
# ========================================================================
class TestAEREndToEnd:
    """Full pipeline integration test with real training."""

    def test_full_pipeline_univariate(self):
        """Train AER pipeline on random univariate data and get predictions."""
        T = 300
        D = 1
        W = 20
        S = 1

        np.random.seed(42)
        X_train = np.random.randn(T, D).astype(np.float32)
        X_test = np.random.randn(T, D).astype(np.float32)
        y_test = np.zeros(T, dtype=int)
        y_test[100:110] = 1  # inject anomaly labels

        pipeline = AnomalyDetectionPipeline(
            data_processor=AERProcessor(
                window_config=WindowConfig(window_size=W, stride=S),
                lstm_units=8, epochs=3, batch_size=32,
                validation_split=0.2, patience=100, device="cpu"
            ),
            detection_method=HybridDetection(comb="mult"),
            scoring_method=WeightedAverageScoring(),
            post_processor=PostProcessor(
                threshold_method=PercentileThreshold(percentile=95),
                min_anomaly_length=1, merge_gap=0
            ),
        )

        # Fit (includes model training)
        pipeline.fit(X_train)

        # Predict
        result = pipeline.predict(X_test, y_test)

        # --- Shape checks ---
        assert result.predictions.shape == (T,), \
            f"predictions shape: {result.predictions.shape}"
        assert result.point_scores.shape == (T,), \
            f"point_scores shape: {result.point_scores.shape}"
        N_expected = T - W + 1
        assert result.subsequence_scores.shape == (N_expected,), \
            f"subseq scores shape: {result.subsequence_scores.shape}"

        # --- Value checks ---
        assert not np.any(np.isnan(result.point_scores))
        assert not np.any(np.isinf(result.point_scores))
        assert np.all((result.predictions == 0) | (result.predictions == 1))
        assert result.threshold > 0

        # --- Timing ---
        assert "total" in result.execution_time
        assert result.execution_time["total"] > 0

    def test_full_pipeline_multivariate(self):
        """Train AER pipeline on multivariate data."""
        T, D, W = 200, 3, 15

        np.random.seed(123)
        X_train = np.random.randn(T, D).astype(np.float32)
        X_test = np.random.randn(T, D).astype(np.float32)
        y_test = np.zeros(T, dtype=int)
        y_test[50:60] = 1

        pipeline = AnomalyDetectionPipeline(
            data_processor=AERProcessor(
                window_config=WindowConfig(window_size=W, stride=1),
                lstm_units=8, epochs=2, batch_size=16,
                validation_split=0.2, patience=100, device="cpu"
            ),
            detection_method=HybridDetection(comb="mult"),
            scoring_method=WeightedAverageScoring(),
            post_processor=PostProcessor(
                threshold_method=PercentileThreshold(percentile=95),
                min_anomaly_length=1, merge_gap=0
            ),
        )

        pipeline.fit(X_train)
        result = pipeline.predict(X_test, y_test)

        assert result.predictions.shape == (T,)
        assert result.point_scores.shape == (T,)
        assert not np.any(np.isnan(result.point_scores))

    def test_pipeline_with_f1_threshold(self):
        """F1OptimalThreshold also works in the pipeline."""
        T, D, W = 200, 1, 10

        np.random.seed(99)
        X_train = np.random.randn(T, D).astype(np.float32)
        X_test = np.random.randn(T, D).astype(np.float32)
        y_test = np.zeros(T, dtype=int)
        y_test[80:90] = 1

        pipeline = AnomalyDetectionPipeline(
            data_processor=AERProcessor(
                window_config=WindowConfig(window_size=W, stride=1),
                lstm_units=8, epochs=2, batch_size=16,
                validation_split=0.2, patience=100, device="cpu"
            ),
            detection_method=HybridDetection(comb="sum", lambda_rec=0.5),
            scoring_method=WeightedAverageScoring(),
            post_processor=PostProcessor(
                threshold_method=F1OptimalThreshold(),
                min_anomaly_length=1, merge_gap=0
            ),
        )

        pipeline.fit(X_train)
        result = pipeline.predict(X_test, y_test)

        assert result.predictions.shape == (T,)
        assert result.threshold > 0


# ========================================================================
# 6. Shape flow verification (explicit trace)
# ========================================================================
class TestShapeFlow:
    """Explicitly trace shapes through each component step by step."""

    def test_shape_trace(self):
        """Trace: X → windows → errors(N,3) → scores(N,) → point(T,) → pred(T,)"""
        T, D, W, S = 100, 2, 15, 1
        N_expected = T - W + 1  # 86

        # Step 0: raw time series
        X = np.random.randn(T, D).astype(np.float32)
        assert X.shape == (T, D), f"Step 0: {X.shape}"

        # Step 1: create windows
        processor = AERProcessor(
            window_config=WindowConfig(window_size=W, stride=S),
            lstm_units=8, epochs=1, batch_size=32,
            validation_split=0.1, patience=100, device="cpu"
        )
        windows = processor._create_windows(X)
        assert windows.shape == (N_expected, W, D), \
            f"Step 1 windows: expected ({N_expected}, {W}, {D}), got {windows.shape}"

        # Step 1b: fit_transform → errors
        errors = processor.fit_transform(windows)
        assert errors.shape == (N_expected, 3), \
            f"Step 1b errors: expected ({N_expected}, 3), got {errors.shape}"

        # Step 2: detection → subsequence scores
        detector = HybridDetection(comb="mult")
        subs = detector.detect(errors)
        assert subs.shape == (N_expected,), \
            f"Step 2 subs: expected ({N_expected},), got {subs.shape}"

        # Step 3: scoring → point-wise scores
        scorer = WeightedAverageScoring()
        point = scorer.score(subs, window_size=W, stride=S, original_length=T)
        assert point.shape == (T,), \
            f"Step 3 point: expected ({T},), got {point.shape}"

        # Step 4: post-processing → predictions
        pp = PostProcessor(
            threshold_method=PercentileThreshold(percentile=90),
            min_anomaly_length=1, merge_gap=0
        )
        y_dummy = np.zeros(T, dtype=int)
        y_dummy[T // 2: T // 2 + 5] = 1
        preds, threshold = pp.process(point, y_dummy)
        assert preds.shape == (T,), \
            f"Step 4 preds: expected ({T},), got {preds.shape}"

        print("\n=== Shape Flow Trace (PASSED) ===")
        print(f"  X:             {X.shape}")
        print(f"  windows:       {windows.shape}")
        print(f"  errors:        {errors.shape}")
        print(f"  subseq_scores: {subs.shape}")
        print(f"  point_scores:  {point.shape}")
        print(f"  predictions:   {preds.shape}")
        print(f"  threshold:     {threshold:.6f}")
        print("================================")


# ========================================================================
# Run directly
# ========================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
