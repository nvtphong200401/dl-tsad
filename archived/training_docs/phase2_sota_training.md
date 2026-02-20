# Phase 2: State-of-the-Art Components (Week 2-3)

## Objectives

Implement state-of-the-art anomaly detection methods by building on the Phase 1 infrastructure. This phase focuses on:

1. ✅ Implementing AER (Auto-Encoder with Regression) - Current SOTA
2. ✅ Implementing Anomaly Transformer - Attention-based SOTA
3. ✅ Adding advanced scoring methods
4. ✅ Integrating real-world benchmark datasets
5. ✅ Implementing VUS-PR metric
6. ✅ Running comprehensive comparisons

**Goal**: By end of Phase 2, achieve F1 > 0.75 on standard benchmarks (SMD, MSL, SMAP), matching or exceeding published results.

---

## Deliverables

### 1. AER: Auto-Encoder with Regression

#### Architecture Overview

**Paper**: Wong et al., IEEE Big Data 2022
**Key Innovation**: Hybrid approach combining reconstruction (autoencoder) and prediction (regressor) with bidirectional scoring.

**Model Components**:
1. BiLSTM Encoder: Processes input window in both directions
2. BiLSTM Decoder: Reconstructs original window
3. LSTM Regressor: Predicts next timestep (forward and backward)
4. Joint Loss: α * reconstruction_loss + (1-α) * prediction_loss

#### 1.1 Data Processor for AER

**File**: `src/pipeline/step1_data_processing.py`

**Add class**:
```python
@dataclass
class AERConfig:
    window_size: int = 100
    stride: int = 1
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1

class AERProcessor(DataProcessor):
    """AER-specific data processing

    Trains BiLSTM encoder-decoder + regressor model.
    Returns concatenated [original, reconstruction, forward_pred, backward_pred]
    for use in HybridDetection.
    """

    def __init__(self, config: AERConfig):
        super().__init__(WindowConfig(config.window_size, config.stride))
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Train AER model and return processed windows"""
        # 1. Create windows
        windows = self._create_windows(X)

        # 2. Build model
        self.model = self._build_model(windows.shape[-1])

        # 3. Train model
        self._train(windows)

        # 4. Generate outputs
        return self._process_windows(windows)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Process windows using trained model"""
        windows = self._create_windows(X)
        return self._process_windows(windows)

    def _build_model(self, input_dim: int):
        """Build BiLSTM encoder-decoder + regressor"""
        return AERModel(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )

    def _train(self, windows: np.ndarray, epochs: int = 50):
        """Train model with joint loss"""
        # Convert to torch dataset
        dataset = TensorDataset(torch.FloatTensor(windows))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)

                # Forward pass
                recon, pred_forward, pred_backward = self.model(x)

                # Reconstruction loss
                recon_loss = F.mse_loss(recon, x)

                # Prediction loss (forward)
                pred_loss_f = F.mse_loss(pred_forward[:, :-1], x[:, 1:])

                # Prediction loss (backward)
                pred_loss_b = F.mse_loss(pred_backward[:, 1:], x[:, :-1])

                # Combined loss
                loss = self.config.alpha * recon_loss + \
                       (1 - self.config.alpha) * (pred_loss_f + pred_loss_b) / 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    def _process_windows(self, windows: np.ndarray) -> np.ndarray:
        """Generate [original, recon, pred_f, pred_b] for detection"""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(windows).to(self.device)
            recon, pred_f, pred_b = self.model(x)

            # Concatenate all for detection step
            processed = torch.cat([
                x.flatten(1),
                recon.flatten(1),
                pred_f.flatten(1),
                pred_b.flatten(1)
            ], dim=1)

        return processed.cpu().numpy()
```

#### 1.2 AER Model (PyTorch)

**File**: `src/models/aer.py`

```python
import torch
import torch.nn as nn

class AERModel(nn.Module):
    """BiLSTM Encoder-Decoder + Regressor for AER"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()

        # Encoder (BiLSTM)
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )

        # Decoder (BiLSTM)
        self.decoder = nn.LSTM(
            hidden_dim * 2, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )

        # Reconstruction output
        self.recon_fc = nn.Linear(hidden_dim * 2, input_dim)

        # Forward predictor (LSTM)
        self.predictor_forward = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.pred_fc_forward = nn.Linear(hidden_dim, input_dim)

        # Backward predictor (LSTM on reversed sequence)
        self.predictor_backward = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.pred_fc_backward = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            recon: (batch, seq_len, input_dim) - Reconstructed sequence
            pred_forward: (batch, seq_len, input_dim) - Forward predictions
            pred_backward: (batch, seq_len, input_dim) - Backward predictions
        """
        batch_size, seq_len, _ = x.shape

        # Encode
        encoded, _ = self.encoder(x)

        # Decode (reconstruct)
        decoded, _ = self.decoder(encoded)
        recon = self.recon_fc(decoded)

        # Forward prediction
        pred_f_hidden, _ = self.predictor_forward(x)
        pred_forward = self.pred_fc_forward(pred_f_hidden)

        # Backward prediction (reverse sequence)
        x_reversed = torch.flip(x, dims=[1])
        pred_b_hidden, _ = self.predictor_backward(x_reversed)
        pred_backward = self.pred_fc_backward(pred_b_hidden)
        pred_backward = torch.flip(pred_backward, dims=[1])  # Flip back

        return recon, pred_forward, pred_backward
```

#### 1.3 Hybrid Detection Method

**File**: `src/pipeline/step2_detection.py`

**Add class**:
```python
class HybridDetection(DetectionMethod):
    """AER-style: Combine reconstruction + prediction + bidirectional

    Expects input format: [original, recon, pred_f, pred_b]
    each of shape (N, W*D)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """
        Args:
            alpha: Weight between reconstruction and prediction errors
            beta: Weight between forward and backward prediction
        """
        self.alpha = alpha
        self.beta = beta

    def fit(self, X_processed: np.ndarray, y: Optional[np.ndarray] = None):
        # No training needed for detection step with AER
        pass

    def detect(self, X_processed: np.ndarray) -> np.ndarray:
        """Compute hybrid anomaly score"""
        # Split input
        n_samples, total_dim = X_processed.shape
        chunk_size = total_dim // 4

        original = X_processed[:, :chunk_size]
        recon = X_processed[:, chunk_size:2*chunk_size]
        pred_f = X_processed[:, 2*chunk_size:3*chunk_size]
        pred_b = X_processed[:, 3*chunk_size:]

        # Reconstruction error
        recon_error = np.mean((original - recon) ** 2, axis=1)

        # Forward prediction error
        pred_error_f = np.mean((original - pred_f) ** 2, axis=1)

        # Backward prediction error
        pred_error_b = np.mean((original - pred_b) ** 2, axis=1)

        # Bidirectional prediction error
        pred_error = self.beta * pred_error_f + (1 - self.beta) * pred_error_b

        # Combined score
        scores = self.alpha * recon_error + (1 - self.alpha) * pred_error

        return scores
```

**Test criteria**:
- ✅ AER model trains without errors
- ✅ Forward/backward predictions have correct shapes
- ✅ Hybrid detection combines errors correctly
- ✅ Achieves F1 > 0.70 on synthetic data

---

### 2. Anomaly Transformer

#### Architecture Overview

**Paper**: Xu et al., ICLR 2022
**Key Innovation**: Association discrepancy - compare learned associations (series-association) with prior associations (Gaussian kernel) to detect anomalies.

#### 2.1 Data Processor for Anomaly Transformer

**File**: `src/pipeline/step1_data_processing.py`

**Add class**:
```python
@dataclass
class AnomalyTransformerConfig:
    window_size: int = 100
    stride: int = 1
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.1

class AnomalyTransformerProcessor(DataProcessor):
    """Anomaly Transformer data processing

    Trains transformer with association discrepancy loss.
    Returns association features for detection.
    """

    def __init__(self, config: AnomalyTransformerConfig):
        super().__init__(WindowConfig(config.window_size, config.stride))
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Train transformer and return processed windows"""
        windows = self._create_windows(X)

        self.model = self._build_model(windows.shape[-1])
        self._train(windows, epochs=10)  # Quick training!

        return self._process_windows(windows)

    def transform(self, X: np.ndarray) -> np.ndarray:
        windows = self._create_windows(X)
        return self._process_windows(windows)

    def _build_model(self, input_dim: int):
        """Build Anomaly Transformer"""
        from src.models.anomaly_transformer import AnomalyTransformer
        return AnomalyTransformer(
            input_dim=input_dim,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout
        )

    def _train(self, windows: np.ndarray, epochs: int = 10):
        """Train with association discrepancy loss"""
        dataset = TensorDataset(torch.FloatTensor(windows))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)

                # Forward pass
                output, series_association, prior_association = self.model(x)

                # Association discrepancy loss
                loss = self.model.compute_loss(output, x, series_association, prior_association)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    def _process_windows(self, windows: np.ndarray) -> np.ndarray:
        """Extract association discrepancy features"""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(windows).to(self.device)
            output, series_assoc, prior_assoc = self.model(x)

            # Return association discrepancy as features
            discrepancy = torch.abs(series_assoc - prior_assoc)

        return discrepancy.cpu().numpy()
```

#### 2.2 Anomaly Transformer Model

**File**: `src/models/anomaly_transformer.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AnomalyTransformer(nn.Module):
    """Anomaly Transformer with Association Discrepancy"""

    def __init__(self, input_dim: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            AnomalyTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            output: (batch, seq_len, input_dim) - Reconstructed
            series_association: (batch, seq_len, seq_len) - Learned associations
            prior_association: (batch, seq_len, seq_len) - Prior (Gaussian kernel)
        """
        batch_size, seq_len, _ = x.shape

        # Project to d_model
        x_proj = self.input_projection(x)

        # Compute prior association (Gaussian kernel)
        prior_association = self._compute_prior_association(seq_len)
        prior_association = prior_association.unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)

        # Pass through transformer layers
        series_associations = []
        for layer in self.transformer_layers:
            x_proj, series_assoc = layer(x_proj)
            series_associations.append(series_assoc)

        # Average series associations across layers
        series_association = torch.stack(series_associations).mean(dim=0)

        # Project back to input dimension
        output = self.output_projection(x_proj)

        return output, series_association, prior_association

    def _compute_prior_association(self, seq_len: int):
        """Compute Gaussian kernel as prior"""
        positions = torch.arange(seq_len).float()
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)) ** 2
        sigma = seq_len / 6  # Hyperparameter
        prior = torch.exp(-distances / (2 * sigma ** 2))
        # Normalize
        prior = prior / prior.sum(dim=-1, keepdim=True)
        return prior

    def compute_loss(self, output, target, series_assoc, prior_assoc, lambda_assoc=1.0):
        """Association discrepancy loss"""
        # Reconstruction loss
        recon_loss = F.mse_loss(output, target)

        # Association discrepancy
        assoc_discrepancy = F.kl_div(
            series_assoc.log(),
            prior_assoc,
            reduction='batchmean'
        )

        # Combined loss (minimize reconstruction, maximize discrepancy in anomalies)
        # During training on normal data, we want to minimize both
        loss = recon_loss + lambda_assoc * assoc_discrepancy

        return loss


class AnomalyTransformerLayer(nn.Module):
    """Single Anomaly Transformer layer"""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, seq_len, seq_len)
        """
        # Self-attention with residual
        attn_output, attn_weights = self.attention(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.norm1(x + attn_output)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        return x, attn_weights
```

#### 2.3 Association Discrepancy Detection

**File**: `src/pipeline/step2_detection.py`

**Add class**:
```python
class AssociationDiscrepancyDetection(DetectionMethod):
    """Anomaly Transformer detection using association discrepancy

    Input is already the discrepancy computed by AnomalyTransformerProcessor
    """

    def fit(self, X_processed: np.ndarray, y: Optional[np.ndarray] = None):
        # No additional training needed
        pass

    def detect(self, X_processed: np.ndarray) -> np.ndarray:
        """Aggregate discrepancy to get anomaly score per window"""
        # X_processed is (N, W, W) - discrepancy matrices
        # Compute score as mean discrepancy per window
        scores = np.mean(X_processed, axis=(1, 2))
        return scores
```

**Test criteria**:
- ✅ Anomaly Transformer trains in < 10 epochs
- ✅ Association discrepancy is computed correctly
- ✅ Detection produces reasonable scores
- ✅ Achieves F1 > 0.70 on synthetic data

---

### 3. Advanced Scoring Methods

**File**: `src/pipeline/step3_scoring.py`

**Add classes**:

```python
class WeightedAverageScoring(ScoringMethod):
    """Gaussian weighted average - center of window gets more weight"""

    def score(self, subsequence_scores, window_size, stride, original_length):
        # Gaussian weights centered at middle
        weights = np.exp(-0.5 * ((np.arange(window_size) - window_size/2) / (window_size/6)) ** 2)
        weights = weights / weights.sum()

        point_scores = np.zeros(original_length)
        weight_sums = np.zeros(original_length)

        for i, score in enumerate(subsequence_scores):
            start = i * stride
            end = start + window_size
            point_scores[start:end] += score * weights
            weight_sums[start:end] += weights

        point_scores = point_scores / np.maximum(weight_sums, 1e-8)
        return point_scores


class GaussianSmoothingScoring(ScoringMethod):
    """Apply Gaussian smoothing after aggregation"""

    def __init__(self, sigma: float = 2.0):
        self.sigma = sigma

    def score(self, subsequence_scores, window_size, stride, original_length):
        from scipy.ndimage import gaussian_filter1d

        # First average pool
        avg_scoring = AveragePoolingScoring()
        point_scores = avg_scoring.score(subsequence_scores, window_size, stride, original_length)

        # Smooth
        smoothed = gaussian_filter1d(point_scores, sigma=self.sigma)
        return smoothed
```

---

### 4. Real-World Datasets

**File**: `src/data/loader.py`

**Add dataset loaders**:

```python
def load_smd_dataset(data_dir: str = "data/SMD") -> Dict[str, Dataset]:
    """Load SMD (Server Machine Dataset)

    SMD contains 28 machines with 38-dimensional multivariate time series.
    Returns dict with keys like "machine-1-1", "machine-1-2", etc.
    """
    datasets = {}

    # SMD structure: train/ and test/ folders with files like machine-1-1.txt
    for machine_file in os.listdir(os.path.join(data_dir, "train")):
        machine_id = machine_file.replace(".txt", "")

        # Load train data
        train_data = np.loadtxt(os.path.join(data_dir, "train", machine_file), delimiter=",")

        # Load test data
        test_data = np.loadtxt(os.path.join(data_dir, "test", machine_file), delimiter=",")

        # Load labels
        label_file = os.path.join(data_dir, "test_label", machine_file)
        test_labels = np.loadtxt(label_file, delimiter=",") if os.path.exists(label_file) else None

        # Split validation from training (last 20%)
        split_idx = int(len(train_data) * 0.8)
        X_train = train_data[:split_idx]
        X_val = train_data[split_idx:]

        datasets[machine_id] = Dataset(
            X_train=X_train,
            y_train=np.zeros(len(X_train)),  # Training is all normal
            X_val=X_val,
            y_val=np.zeros(len(X_val)),
            X_test=test_data,
            y_test=test_labels if test_labels is not None else np.zeros(len(test_data)),
            name=f"SMD-{machine_id}",
            metadata={"source": "SMD", "n_dims": train_data.shape[1]}
        )

    return datasets


def load_msl_smap_dataset(dataset_name: str, data_dir: str = "data/") -> Dataset:
    """Load MSL or SMAP dataset

    NASA datasets from telemetry data.
    """
    assert dataset_name in ["MSL", "SMAP"]

    data_path = os.path.join(data_dir, dataset_name)

    # Load train data
    train_data = np.load(os.path.join(data_path, "train.npy"))

    # Load test data
    test_data = np.load(os.path.join(data_path, "test.npy"))

    # Load labels
    test_labels = np.load(os.path.join(data_path, "test_label.npy"))

    # Split validation
    split_idx = int(len(train_data) * 0.8)

    return Dataset(
        X_train=train_data[:split_idx],
        y_train=np.zeros(split_idx),
        X_val=train_data[split_idx:],
        y_val=np.zeros(len(train_data) - split_idx),
        X_test=test_data,
        y_test=test_labels,
        name=dataset_name,
        metadata={"source": dataset_name, "n_dims": train_data.shape[1]}
    )


def download_datasets():
    """Helper to download benchmark datasets"""
    # Implementation to download from public repos
    # SMD: https://github.com/NetManAIOps/OmniAnomaly
    # MSL/SMAP: https://github.com/khundman/telemanom
    pass
```

**Dataset preparation script**: `scripts/download_datasets.sh`
```bash
#!/bin/bash

# Download SMD
mkdir -p data/SMD
wget https://github.com/NetManAIOps/OmniAnomaly/raw/master/ServerMachineDataset/...

# Download MSL/SMAP
mkdir -p data/MSL data/SMAP
wget https://github.com/khundman/telemanom/raw/master/data/...

echo "Datasets downloaded successfully!"
```

---

### 5. VUS-PR Metric

**File**: `src/evaluation/metrics.py`

**Add function**:
```python
def compute_vus_pr(y_true: np.ndarray, scores: np.ndarray, num_thresholds: int = 100) -> float:
    """Volume Under Surface for Precision-Recall curve

    More reliable than point-wise metrics according to TSB-AD benchmark.

    Args:
        y_true: Binary labels (T,)
        scores: Anomaly scores (T,)
        num_thresholds: Number of thresholds to evaluate

    Returns:
        VUS-PR score (higher is better)
    """
    from sklearn.metrics import precision_recall_curve, auc

    # Get range of thresholds
    thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)

    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)

        # Compute precision and recall
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    # Sort by recall for proper AUC calculation
    recalls, precisions = zip(*sorted(zip(recalls, precisions)))

    # Compute area under PR curve
    vus_pr = auc(recalls, precisions)

    return vus_pr
```

---

### 6. Comprehensive Evaluation Script

**File**: `experiments/compare_sota.py`

```python
#!/usr/bin/env python
"""Compare SOTA methods on benchmark datasets"""

import yaml
import numpy as np
import pandas as pd
from src.utils.config_factory import build_pipeline_from_config
from src.data.loader import load_smd_dataset, load_msl_smap_dataset
from src.evaluation.evaluator import Evaluator

def run_comparison():
    """Run comprehensive comparison"""

    # Pipelines to compare
    pipeline_configs = [
        "configs/pipelines/baseline_knn.yaml",
        "configs/pipelines/aer_pipeline.yaml",
        "configs/pipelines/transformer_pipeline.yaml"
    ]

    # Datasets to evaluate
    datasets = {
        "MSL": load_msl_smap_dataset("MSL"),
        "SMAP": load_msl_smap_dataset("SMAP"),
        # Can add SMD machines individually
    }

    results = []

    for config_path in pipeline_configs:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        pipeline_name = config['experiment']['name']
        print(f"\n{'='*60}")
        print(f"Evaluating: {pipeline_name}")
        print(f"{'='*60}")

        for dataset_name, dataset in datasets.items():
            print(f"\nDataset: {dataset_name}")

            # Build pipeline
            pipeline = build_pipeline_from_config(config)

            # Train
            print("  Training...")
            pipeline.fit(dataset.X_train, dataset.y_train)

            # Evaluate on validation for threshold tuning
            print("  Tuning threshold...")
            val_result = pipeline.predict(dataset.X_val, dataset.y_val)

            # Evaluate on test
            print("  Testing...")
            test_result = pipeline.predict(dataset.X_test, dataset.y_test)

            # Compute metrics
            evaluator = Evaluator()
            eval_result = evaluator.evaluate(
                y_true=dataset.y_test,
                y_pred=test_result.predictions,
                scores=test_result.point_scores
            )

            # Store results
            results.append({
                'Pipeline': pipeline_name,
                'Dataset': dataset_name,
                'F1': eval_result.f1,
                'Precision': eval_result.precision,
                'Recall': eval_result.recall,
                'PA-F1': eval_result.pa_f1,
                'VUS-PR': eval_result.vus_pr,
                'Latency (ms)': eval_result.latency_ms
            })

            print(f"  F1: {eval_result.f1:.3f}, PA-F1: {eval_result.pa_f1:.3f}, VUS-PR: {eval_result.vus_pr:.3f}")

    # Create results table
    df_results = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(df_results.to_string(index=False))

    # Save to CSV
    df_results.to_csv("experiments/results_sota_comparison.csv", index=False)
    print("\nResults saved to experiments/results_sota_comparison.csv")

    # Print best performing method per dataset
    print(f"\n{'='*60}")
    print("BEST METHODS PER DATASET")
    print(f"{'='*60}")
    for dataset in df_results['Dataset'].unique():
        subset = df_results[df_results['Dataset'] == dataset]
        best = subset.loc[subset['F1'].idxmax()]
        print(f"{dataset}: {best['Pipeline']} (F1={best['F1']:.3f})")

if __name__ == "__main__":
    run_comparison()
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/test_aer.py`
- Test AER model forward pass
- Test training loop
- Test hybrid detection
- Test with different hyperparameters

**File**: `tests/test_anomaly_transformer.py`
- Test Anomaly Transformer forward pass
- Test association discrepancy computation
- Test training convergence
- Test detection

**File**: `tests/test_datasets.py`
- Test dataset loading
- Test data format
- Test train/val/test splits
- Test with real data files

### Integration Tests

**File**: `tests/test_sota_pipelines.py`
- Test AER end-to-end
- Test Anomaly Transformer end-to-end
- Compare against baseline
- Verify performance improvements

---

## Development Timeline

### Week 2 (Days 8-14): AER Implementation
- [ ] Day 8-9: Implement AER model architecture
- [ ] Day 10: Implement AERProcessor
- [ ] Day 11: Implement HybridDetection
- [ ] Day 12: Test on synthetic data
- [ ] Day 13: Test on real datasets (MSL/SMAP)
- [ ] Day 14: Tune hyperparameters, document

### Week 3 (Days 15-21): Anomaly Transformer + Evaluation
- [ ] Day 15-16: Implement Anomaly Transformer architecture
- [ ] Day 17: Implement AnomalyTransformerProcessor
- [ ] Day 18: Implement AssociationDiscrepancyDetection
- [ ] Day 19: Add VUS-PR metric
- [ ] Day 20: Run comprehensive comparison
- [ ] Day 21: Analysis, documentation, Phase 2 report

---

## Success Criteria for Phase 2

1. ✅ **AER achieves F1 > 0.70** on MSL/SMAP
   - Should match or exceed paper's reported 0.753 average

2. ✅ **Anomaly Transformer achieves F1 > 0.70** on MSL/SMAP
   - Should be competitive with AER

3. ✅ **VUS-PR metric implemented correctly**
   - Matches TSB-AD implementation

4. ✅ **Comparison table generated**
   - Shows relative performance of all methods
   - Identifies best method per dataset

5. ✅ **Ready for Phase 3 optimization**
   - Baseline and SOTA methods working
   - Evaluation framework complete
   - Can now experiment with combinations

---

## Configuration Examples

**File**: `configs/pipelines/aer_pipeline.yaml`
```yaml
experiment:
  name: "AER"
  description: "Auto-Encoder with Regression"

data_processing:
  type: "AERProcessor"
  window_size: 100
  stride: 1
  params:
    hidden_dim: 128
    num_layers: 2
    dropout: 0.1
    alpha: 0.5  # Weight for reconstruction vs prediction

detection:
  type: "HybridDetection"
  params:
    alpha: 0.5  # Weight for reconstruction vs prediction
    beta: 0.5   # Weight for forward vs backward

scoring:
  type: "WeightedAverageScoring"

postprocessing:
  threshold:
    type: "F1OptimalThreshold"
  min_anomaly_length: 3
  merge_gap: 5
```

**File**: `configs/pipelines/transformer_pipeline.yaml`
```yaml
experiment:
  name: "AnomalyTransformer"
  description: "Transformer with Association Discrepancy"

data_processing:
  type: "AnomalyTransformerProcessor"
  window_size: 100
  stride: 1
  params:
    d_model: 512
    n_heads: 8
    n_layers: 3
    dropout: 0.1

detection:
  type: "AssociationDiscrepancyDetection"

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

---

## Deliverables Checklist

### Code
- [ ] `src/models/aer.py` - AER PyTorch implementation
- [ ] `src/models/anomaly_transformer.py` - Anomaly Transformer
- [ ] `src/pipeline/step1_data_processing.py` - Add AER and Transformer processors
- [ ] `src/pipeline/step2_detection.py` - Add Hybrid and AssociationDiscrepancy
- [ ] `src/pipeline/step3_scoring.py` - Add WeightedAverage and GaussianSmoothing
- [ ] `src/data/loader.py` - Add SMD, MSL, SMAP loaders
- [ ] `src/evaluation/metrics.py` - Add VUS-PR
- [ ] `experiments/compare_sota.py` - Comparison script

### Configs
- [ ] `configs/pipelines/aer_pipeline.yaml`
- [ ] `configs/pipelines/transformer_pipeline.yaml`

### Tests
- [ ] `tests/test_aer.py`
- [ ] `tests/test_anomaly_transformer.py`
- [ ] `tests/test_datasets.py`
- [ ] `tests/test_sota_pipelines.py`

### Documentation
- [ ] Phase 2 completion report
- [ ] Performance comparison table
- [ ] Instructions for downloading datasets

---

## Notes

- **GPU required**: Both AER and Anomaly Transformer benefit significantly from GPU
- **Training time**: AER ~30min, Anomaly Transformer ~10min per dataset on GPU
- **Hyperparameter tuning**: Use validation set to tune window_size, hidden_dim, etc.
- **Memory**: 16GB+ GPU recommended for large datasets