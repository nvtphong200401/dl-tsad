"""SOTA Data Processors - AER and Anomaly Transformer"""

import numpy as np
from .step1_data_processing import DataProcessor, WindowConfig


class AERProcessor(DataProcessor):
    """AER-specific data processing (matches Orion reference)

    Trains a single encoder-decoder model that simultaneously
    reconstructs the input and predicts adjacent timesteps.
    Returns per-window error features [rec_error, reg_b_error, reg_f_error]
    for use in HybridDetection.
    """

    def __init__(self,
                 window_config: WindowConfig,
                 lstm_units: int = 30,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 reg_ratio: float = 0.5,
                 epochs: int = 35,
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 validation_split: float = 0.2,
                 patience: int = 10,
                 device: str = None,
                 # Legacy aliases -------
                 hidden_dim: int = None,
                 alpha: float = None):
        super().__init__(window_config)
        # Support legacy config keys
        self.lstm_units = hidden_dim if hidden_dim is not None else lstm_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.reg_ratio = alpha if alpha is not None else reg_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.patience = patience
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
        print(f"    Architecture: BiLSTM encoder ({self.lstm_units} units) → RepeatVector → BiLSTM decoder")

        # Build model (single encoder-decoder, matching Orion reference)
        self.model = AERModel(
            input_dim=D,
            lstm_units=self.lstm_units,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # Split train / val for early stopping
        n_val = max(1, int(N * self.validation_split))
        n_train = N - n_val
        perm = np.random.permutation(N)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        train_dataset = TensorDataset(torch.FloatTensor(windows[train_idx]))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_data = torch.FloatTensor(windows[val_idx]).to(self.device)

        # Train with early stopping
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in train_loader:
                x_full = batch[0].to(self.device)       # (B, W, D)
                x_trimmed = x_full[:, 1:-1, :]           # (B, W-2, D) — trim first & last

                ry, y, fy = self.model(x_trimmed)
                loss = self.model.compute_loss(x_full, ry, y, fy, self.reg_ratio)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                val_trimmed = val_data[:, 1:-1, :]
                v_ry, v_y, v_fy = self.model(val_trimmed)
                val_loss = self.model.compute_loss(val_data, v_ry, v_y, v_fy, self.reg_ratio).item()
            self.model.train()

            # Early stopping check
            if val_loss < best_val_loss - 0.0003:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"      Epoch {epoch+1}/{self.epochs}, Train: {avg_loss:.4f}, Val: {val_loss:.4f}")

            if patience_counter >= self.patience:
                print(f"      Early stopping at epoch {epoch+1} (patience={self.patience})")
                break

        print(f"    AER training complete!")

        # Generate processed windows
        return self._process_windows(windows)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Transform windows using trained model"""
        return self._process_windows(windows)

    def _process_windows(self, windows: np.ndarray) -> np.ndarray:
        """Compute per-window error components for HybridDetection.

        Returns:
            (N, 3) array — [reconstruction_error, backward_reg_error, forward_reg_error]
        """
        import torch

        self.model.eval()
        with torch.no_grad():
            x_full = torch.FloatTensor(windows).to(self.device)  # (N, W, D)
            x_trimmed = x_full[:, 1:-1, :]                       # (N, W-2, D)

            ry, y, fy = self.model(x_trimmed)

            # Per-window reconstruction error
            rec_error = torch.mean((x_trimmed - y) ** 2, dim=(1, 2))  # (N,)

            # Per-window backward regression error
            reg_error_b = torch.mean((x_full[:, 0] - ry) ** 2, dim=1)  # (N,)

            # Per-window forward regression error
            reg_error_f = torch.mean((x_full[:, -1] - fy) ** 2, dim=1)  # (N,)

            errors = torch.stack([rec_error, reg_error_b, reg_error_f], dim=1)  # (N, 3)

        return errors.cpu().numpy()

    def get_output_dim(self) -> int:
        """Output dimension is 3 (rec_error, reg_b_error, reg_f_error)"""
        return 3


class AnomalyTransformerProcessor(DataProcessor):
    """Anomaly Transformer data processing.

    Faithfully implements the minimax training strategy and association
    discrepancy scoring from Xu et al., ICLR 2022.
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
        """Train transformer with minimax strategy and return anomaly scores."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        from ..models.anomaly_transformer import (
            AnomalyTransformer,
            compute_association_discrepancy,
        )

        N, W, D = windows.shape
        self.input_dim = D
        win_size = W

        print(f"    Training Anomaly Transformer (device: {self.device})...")

        # Build model — now requires win_size as first argument
        self.model = AnomalyTransformer(
            win_size=win_size,
            input_dim=D,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Prepare dataset
        dataset = TensorDataset(torch.FloatTensor(windows))
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        # Train with minimax strategy (from original paper)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        k = self.lambda_assoc

        self.model.train()
        for epoch in range(self.epochs):
            total_loss1 = 0.0
            for batch in dataloader:
                x = batch[0].to(self.device)

                # Forward
                output, series_list, prior_list = self.model(x)

                # Reconstruction loss
                rec_loss = criterion(output, x)

                # Association discrepancy (minimax)
                series_loss, prior_loss = compute_association_discrepancy(
                    series_list, prior_list, win_size)

                # Phase 1: minimize rec_loss - k * series_loss
                loss1 = rec_loss - k * series_loss
                # Phase 2: minimize rec_loss + k * prior_loss
                loss2 = rec_loss + k * prior_loss

                optimizer.zero_grad()
                loss1.backward(retain_graph=True)
                loss2.backward()
                optimizer.step()

                total_loss1 += loss1.item()

            avg_loss = total_loss1 / len(dataloader)
            print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        print(f"    Anomaly Transformer training complete!")

        # Generate anomaly scores for windows
        return self._process_windows(windows)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Transform windows using trained model"""
        return self._process_windows(windows)

    def _process_windows(self, windows: np.ndarray) -> np.ndarray:
        """Compute per-timestep anomaly scores using association discrepancy.

        Returns (N, W) array: one anomaly score per timestep per window.
        """
        import torch
        from ..models.anomaly_transformer import compute_anomaly_score

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(windows).to(self.device)
            output, series_list, prior_list = self.model(x)

            # (B, L) anomaly scores
            scores = compute_anomaly_score(
                x, output, series_list, prior_list,
                win_size=windows.shape[1],
                temperature=50.0,
            )

        return scores  # (N, W)

    def get_output_dim(self) -> int:
        """Output dimension is window_size (one score per timestep)."""
        return self.window_config.window_size

