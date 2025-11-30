import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Literal, Optional

class Support(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super(Support, self).__init__()
        if hidden_dim is None:
            hidden_dim = max(16, min(64, input_dim * 2))

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.layers(x).squeeze(-1)

class Train:
    def __init__(
            self,
            train_loader,
            val_loader,
            model: Optional[nn.Module] | None = None,
            epochs: int | None = 100,
            train_patience: int| None = 10,
            device="cpu",
            verbose: int | None = 0,
            loss: Literal['mse', 'mae', 'huber'] | None = 'mse',
            learning_rate: float | None = 1e-3,
            huber_threshold: float | None = 0.5,
            optimizer: Literal['adam', 'adamw'] | None = 'adam',
            weight_decay: float | None = 1e-5,
            factor: float | None = 0.5,
            scheduler_patience: int | None = 5,
            early_stopping: bool | None = True
            ):
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_patience = train_patience
        self.device = device
        self.verbose = verbose
        self.loss = loss
        self.lr_rate = learning_rate
        self.beta = huber_threshold
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.factor = factor
        self.scheduler_patience = scheduler_patience
        self.early_stop = early_stopping

    def train_model(self):
        # Move model to device
        self.model.to(self.device)

        # Setup loss function
        if self.loss == 'mse':
            criterion = nn.MSELoss()
        elif self.loss == 'mae':
            criterion = nn.L1Loss()
        elif self.loss == 'huber':
            criterion = nn.SmoothL1Loss(beta=self.beta)
        else:
            raise ValueError(f'Unexpected loss function: {self.loss}')

        # Setup optimizer
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Unexpected optimizer: {self.optimizer}')

        # Setup scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.factor, patience=self.scheduler_patience
        )

        best_val_loss = float('inf')
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}
        best_model_wts = self.model.state_dict().copy()

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []

            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            val_losses = []
            self.model.eval()
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    outputs = self.model(xb)
                    loss = criterion(outputs, yb)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            if self.verbose == 1:
                print(f"|=Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}=|")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_wts = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= self.train_patience and self.early_stop:
                if self.verbose == 1:
                    print(f"|=Stopping estimator training early at epoch {epoch + 1}=|")
                break

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model, history  # Return both model and history

    def predict(self, X):
        """Make predictions using the trained model"""
        self.model.eval()
        with torch.no_grad():
            if not torch.is_tensor(X):
                X = torch.tensor(X, dtype=torch.float32, device=self.device)
            elif X.device != self.device:
                X = X.to(self.device)

            predictions = self.model(X)
            return predictions.cpu().numpy() if predictions.is_cuda else predictions.numpy()