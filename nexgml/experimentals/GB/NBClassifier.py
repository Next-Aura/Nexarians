import numpy as np
from typing import Optional, Literal, Any
from scipy.sparse import spmatrix, issparse, csr_matrix
from math import sqrt, log2
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn

from .childs.NBC_support import Train, Support

class NBClassifier:
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        n_estimators: int = 50,
        max_features: Optional[int | str | float] = 'sqrt',
        epochs: int = 50,
        train_patience: int = 10,
        device: str = "cpu",
        verbose: int = 0,
        shuffle: bool = True,
        loss: Literal['cross_entropy'] = 'cross_entropy',
        learning_rate: float = 0.1,
        optimizer: Literal['adam', 'adamw'] = 'adamw',
        weight_decay: float = 1e-5,
        factor: float = 0.5,
        penalty: Literal["l1", "l2", "elasticnet"] = "l2",
        scheduler_patience: int = 5,
        early_stopping: bool = True,
        l1_ratio: float = 0.5,
        random_state: Any = None,
        bootstrap: bool = True,
        tol: float = 1e-4,
        batch_size: int = 32
    ):
        self.model = model
        self.n_estimators = n_estimators
        self.max_iter = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.tol = tol
        self.loss = loss
        self.early_stop = early_stopping
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.num_classes = None
        self.optimizer = optimizer
        self.factor = factor
        self.scheduler_patience = scheduler_patience
        self.shuffle = shuffle
        self.train_patience = train_patience
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.weight_decay = weight_decay
        self.max_features = max_features
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.loss_history = []
        self.supports = []
        self.feature_indices = []
        self.model_data = []
        self.init_prediction = None

        if self.n_estimators < 1:
            raise ValueError("Number of estimators must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")

    def _handle_sparse_matrix_indexing(self, X: np.ndarray | spmatrix, indices: np.ndarray) -> np.ndarray:
        """Handle sparse matrix indexing properly"""
        if issparse(X):
            X = X.tocsr()
            return X[indices].toarray()
        return X[indices]

    def _handle_sparse_matrix_column_selection(self, X: np.ndarray | spmatrix, feature_idx: np.ndarray) -> np.ndarray:
        """Handle sparse matrix column selection efficiently"""
        if issparse(X):
            X = X.tocsr()
            return X[:, feature_idx].toarray()
        return X[:, feature_idx]

    def fit(self, X: np.ndarray | spmatrix, y: np.ndarray) -> 'NBClassifier':
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        if not issparse(X):
            X = np.asarray(X)
        y = np.asarray(y)

        self.num_classes = len(np.unique(y))

        class_counts = np.bincount(y, minlength=self.num_classes)
        weights = 1.0 / (class_counts + 1e-7)  # Inverse frequency
        weights = weights / weights.sum() * self.num_classes  # Normalisasi
        class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        n_samples, n_features = X.shape
        if len(y) != n_samples:
            raise ValueError(f"Number of samples in X ({n_samples}) and y ({len(y)}) must be equal")
        if len(np.unique(y)) != self.num_classes:
            raise ValueError(f"Number of unique classes in y ({len(np.unique(y))}) must match num_classes ({self.num_classes})")

        # Determine max_features
        if self.max_features is None:
            max_features = n_features
        elif isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                max_features = max(1, int(sqrt(n_features)))
            elif self.max_features == 'log2':
                max_features = max(1, int(log2(n_features)))
            else:
                raise ValueError(f"Unexpected max_features string value: {self.max_features}")
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = max(1, int(np.ceil(self.max_features * n_features)))
        else:
            raise ValueError(f"Unexpected max_features type: {type(self.max_features)}")
        max_features = min(max_features, n_features)

        # Initialize predictions
        y_targets = y.astype(np.int64)
        class_counts = np.bincount(y, minlength=self.num_classes)
        self.init_prediction = np.log((class_counts + 1e-7) / (n_samples + 1e-7 * self.num_classes))
        self.init_prediction = np.tile(self.init_prediction, (n_samples, 1))
        current_predictions = self.init_prediction
        for i in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
            else:
                indices = np.arange(n_samples)

            # Random feature selection
            feature_idx = np.random.choice(n_features, size=max_features, replace=False)
            self.feature_indices.append(feature_idx)

            # Split into train and validation
            train_size = int(0.8 * len(indices))
            idx_perm = np.random.permutation(len(indices))
            train_idx = idx_perm[:train_size]
            val_idx = idx_perm[train_size:]

            # Prepare data
            X_sub = self._handle_sparse_matrix_indexing(X, indices)
            y_sub = y_targets[indices]
            X_sub = self._handle_sparse_matrix_column_selection(X_sub, feature_idx)

            # Convert to tensors
            X_train_tensor = torch.tensor(X_sub[train_idx], dtype=torch.float32, device=self.device)
            y_train_tensor = torch.tensor(y_sub[train_idx], dtype=torch.long, device=self.device)
            X_val_tensor = torch.tensor(X_sub[val_idx], dtype=torch.float32, device=self.device)
            y_val_tensor = torch.tensor(y_sub[val_idx], dtype=torch.long, device=self.device)

            # Create data loaders
            train_ds = TensorDataset(X_train_tensor, y_train_tensor)
            val_ds = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_ds, batch_size=min(self.batch_size, len(train_ds)), shuffle=self.shuffle)
            val_loader = DataLoader(val_ds, batch_size=min(self.batch_size, len(val_ds)), shuffle=False)

            # Initialize model
            support_model = Support(input_dim=max_features, num_classes=self.num_classes) if self.model is None else self.model(input_dim=max_features, num_classes=self.num_classes)
            support_model.to(self.device)

            # Train model
            support = Train(
                train_loader=train_loader,
                val_loader=val_loader,
                model=support_model,
                learning_rate=self.learning_rate,
                loss=self.loss,
                early_stopping=self.early_stop,
                verbose=self.verbose,
                num_classes=self.num_classes,
                optimizer=self.optimizer,
                device=self.device,
                weight_decay=self.weight_decay,
                scheduler_patience=self.scheduler_patience,
                train_patience=self.train_patience,
                epochs=self.max_iter,
                factor=self.factor,
                class_weight=class_weights
            )

            if self.verbose == 2:
                print(f"|=-Training {i + 1}th model-=|")

            model_data = support.train_model()
            self.supports.append(support)
            self.model_data.append(model_data)

            # Get validation loss
            history = model_data[1]
            final_val_loss = history['val_loss'][-1] if 'val_loss' in history and history['val_loss'] else float('inf')
            self.loss_history.append(final_val_loss)

            # Update predictions
            X_full_sub = self._handle_sparse_matrix_column_selection(X, feature_idx)
            X_full_tensor = torch.tensor(X_full_sub, dtype=torch.float32, device=self.device)
            pred_full = support.predict(X_full_tensor)
            pred_full = np.clip(pred_full, 1e-7, 1.0 - 1e-7)  # Clip predictions for stability
            current_predictions += self.learning_rate * pred_full

            if self.verbose == 1:
                print(f"|=Trained {i+1}/{self.n_estimators}th model, Loss: {self.loss_history[-1]:.4f}=|")

            # Early stopping
            if self.early_stop and i > 2:
                recent_losses = self.loss_history[-3:]
                if max(recent_losses) - min(recent_losses) < self.tol:
                    if self.verbose == 1:
                        print(f"|=Stopping supports training at estimator {i + 1}=|")
                    break

        return self

    def predict(self, X: np.ndarray | spmatrix) -> np.ndarray:
        if self.init_prediction is None:
            raise ValueError("Model has not been fitted yet")

        if not issparse(X):
            X = np.asarray(X)

        n_samples = X.shape[0]
        final_predictions = np.zeros((n_samples, self.num_classes), dtype=np.float32)

        for i, support in enumerate(self.supports):
            feature_idx = self.feature_indices[i]
            X_sub = self._handle_sparse_matrix_column_selection(X, feature_idx)
            X_sub_tensor = torch.tensor(X_sub, dtype=torch.float32, device=self.device)
            pred = support.predict(X_sub_tensor)
            pred = np.clip(pred, 1e-7, 1.0 - 1e-7)  # Clip predictions for stability
            final_predictions += self.learning_rate * pred

        # Convert to class labels
        return np.argmax(final_predictions, axis=1)

    def get_params(self, deep: bool = True) -> dict:
        return {
            'n_estimators': self.n_estimators,
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate,
            'verbose': self.verbose,
            'penalty': self.penalty,
            'l1_ratio': self.l1_ratio,
            'loss': self.loss,
            'random_state': self.random_state,
            'early_stopping': self.early_stop,
            'bootstrap': self.bootstrap,
            'max_features': self.max_features,
            'num_classes': self.num_classes,
            'train_patience': self.train_patience,
            'scheduler_patience': self.scheduler_patience,
            'weight_decay': self.weight_decay,
            'optimizer': self.optimizer,
            'factor': self.factor,
            'device': self.device,
            'shuffle': self.shuffle,
            'tol': self.tol,
            'batch_size': self.batch_size,
            'model': self.model
        }

    def set_params(self, **params) -> 'NBClassifier':
        for key, value in params.items():
            setattr(self, key, value)
        return self