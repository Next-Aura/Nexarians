import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Literal, Optional

class Support(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, num_classes: int = 2):
        super(Support, self).__init__()
        if hidden_dim is None:
            hidden_dim = max(16, min(64, input_dim * 2))

        # Simplified architecture for efficiency
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for regularization
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Softmax(dim=1)  # Softmax for multi-class probabilities
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is float32 to prevent type mismatch
        x = x.to(torch.float32)
        return self.layers(x)

class Train:
    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model: Optional[nn.Module] = None,
        epochs: int = 100,
        train_patience: int = 10,
        device: str = "cpu",
        verbose: int = 0,
        loss: Literal['cross_entropy'] = 'cross_entropy',
        learning_rate: float = 1e-3,
        num_classes: int | None = None,
        optimizer: Literal['adam', 'adamw'] = 'adam',
        weight_decay: float = 1e-5,
        factor: float = 0.5,
        scheduler_patience: int = 5,
        early_stopping: bool = True,
        class_weight: bool | None = None
    ):
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_patience = train_patience
        self.device = torch.device(device)
        self.verbose = verbose
        self.loss = loss
        self.lr_rate = learning_rate
        self.num_classes = num_classes
        self.optimizer_type = optimizer
        self.weight_decay = weight_decay
        self.factor = factor
        self.scheduler_patience = scheduler_patience
        self.early_stop = early_stopping
        self.class_weight = class_weight

        # Validate inputs
        if self.num_classes < 2:
            raise ValueError("Number of classes must be at least 2")
        if self.lr_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")

    def train_model(self):
        # Move model to device
        self.model.to(self.device)

        # Setup loss function
        if self.loss == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(weight=self.class_weight)
        else:
            raise ValueError(f'Unexpected loss function: {self.loss}')

        # Setup optimizer
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Unexpected optimizer: {self.optimizer_type}')

        # Setup scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.factor, patience=self.scheduler_patience
        )

        best_val_loss = float('inf')
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}
        best_model_wts = self.model.state_dict()

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []

            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                yb = yb.long()  # Ensure labels are long for CrossEntropyLoss

                optimizer.zero_grad()
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                loss.backward()

                # Gradient clipping to prevent overflow
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            val_losses = []
            self.model.eval()
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    yb = yb.long()
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
                best_model_wts = self.model.state_dict()
            else:
                patience_counter += 1

            if self.early_stop and patience_counter >= self.train_patience:
                if self.verbose == 1:
                    print(f"|=Stopping estimator training early at epoch {epoch + 1}=|")
                break

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model, history

    def predict(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        self.model.eval()
        with torch.no_grad():
            if not torch.is_tensor(X):
                X = torch.tensor(X, dtype=torch.float32, device=self.device)
            elif X.device != self.device:
                X = X.to(self.device)

            predictions = self.model(X)
            # Clip predictions to prevent numerical instability
            predictions = torch.clamp(predictions, min=1e-7, max=1.0 - 1e-7)
            return predictions.cpu().numpy()