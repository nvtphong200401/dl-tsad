"""SOTA Data Processors - AER and Anomaly Transformer"""

import numpy as np
from .step1_data_processing import DataProcessor, WindowConfig


class AERProcessor(DataProcessor):
    """AER-specific data processing

    Trains BiLSTM encoder-decoder + regressor model.
    Returns concatenated [original, reconstruction, forward_pred, backward_pred]
    for use in HybridDetection.
    """

    def __init__(self,
                 window_config: WindowConfig,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 alpha: float = 0.5,
                 epochs: int = 50,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 device: str = None):
        super().__init__(window_config)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else ('cuda' if self._check_cuda() else 'cpu')
        self.model = None
        self.input_dim = None

    def _check_cuda(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """Train AER model and return processed windows"""
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        from ..models.aer import AERModel

        N, W, D = windows.shape
        self.input_dim = D

        print(f"    Training AER model (device: {self.device})...")

        # Build model
        self.model = AERModel(
            input_dim=D,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # Prepare dataset
        dataset = TensorDataset(torch.FloatTensor(windows))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)

                # Forward pass
                recon, pred_forward, pred_backward = self.model(x)

                # Compute loss
                loss = self.model.compute_loss(x, recon, pred_forward, pred_backward, self.alpha)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        print(f"    AER training complete!")

        # Generate processed windows
        return self._process_windows(windows)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Transform windows using trained model"""
        return self._process_windows(windows)

    def _process_windows(self, windows: np.ndarray) -> np.ndarray:
        """Generate [original, recon, pred_f, pred_b] for detection"""
        import torch

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(windows).to(self.device)
            recon, pred_f, pred_b = self.model(x)

            # Flatten and concatenate all for detection step
            processed = torch.cat([
                x.flatten(1),
                recon.flatten(1),
                pred_f.flatten(1),
                pred_b.flatten(1)
            ], dim=1)

        return processed.cpu().numpy()

    def get_output_dim(self) -> int:
        """Output dimension is 4 * window_size * input_dim"""
        if self.input_dim:
            return 4 * self.window_config.window_size * self.input_dim
        return 0


class AnomalyTransformerProcessor(DataProcessor):
    """Anomaly Transformer data processing

    Trains transformer with association discrepancy loss.
    Returns association discrepancy features for detection.
    """

    def __init__(self,
                 window_config: WindowConfig,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 dropout: float = 0.1,
                 lambda_assoc: float = 1.0,
                 epochs: int = 10,
                 batch_size: int = 32,
                 learning_rate: float = 1e-4,
                 device: str = None):
        super().__init__(window_config)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lambda_assoc = lambda_assoc
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else ('cuda' if self._check_cuda() else 'cpu')
        self.model = None
        self.input_dim = None

    def _check_cuda(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """Train transformer and return processed windows"""
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        from ..models.anomaly_transformer import AnomalyTransformer

        N, W, D = windows.shape
        self.input_dim = D

        print(f"    Training Anomaly Transformer (device: {self.device})...")

        # Build model
        self.model = AnomalyTransformer(
            input_dim=D,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout
        ).to(self.device)

        # Prepare dataset
        dataset = TensorDataset(torch.FloatTensor(windows))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)

                # Forward pass
                output, series_association, prior_association = self.model(x)

                # Compute loss
                loss = self.model.compute_loss(
                    output, x,
                    series_association, prior_association,
                    self.lambda_assoc
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        print(f"    Anomaly Transformer training complete!")

        # Generate processed windows
        return self._process_windows(windows)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Transform windows using trained model"""
        return self._process_windows(windows)

    def _process_windows(self, windows: np.ndarray) -> np.ndarray:
        """Extract association discrepancy features"""
        import torch

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(windows).to(self.device)
            output, series_assoc, prior_assoc = self.model(x)

            # Return association discrepancy as features
            # Shape: (N, W, W) -> (N, W*W)
            discrepancy = torch.abs(series_assoc - prior_assoc)
            discrepancy_flat = discrepancy.flatten(1)

        return discrepancy_flat.cpu().numpy()

    def get_output_dim(self) -> int:
        """Output dimension is window_size * window_size"""
        return self.window_config.window_size * self.window_config.window_size
