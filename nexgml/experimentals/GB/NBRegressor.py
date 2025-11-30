import numpy as np
from typing import Optional, Literal, Any
from scipy.sparse import spmatrix, issparse, csr_matrix
from math import sqrt, log2
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn

from .childs.NBR_support import Train, Support

class NBRegressor:
    def __init__(
            self,
            model: Optional[nn.Module] | None = None,
            n_estimators: int | None = 50,
            max_features: Optional[int | str | float] = 'sqrt',
            epochs: int | None = 50,
            train_patience: int| None = 10,
            device="cpu",
            verbose: int | None = 0,
            shuffle: bool | None = True,
            loss: Literal['mse', 'mae', 'huber'] | None = 'mse',
            learning_rate: float | None = 0.1,
            huber_threshold: float | None = 0.5,
            optimizer: Literal['adam', 'adamw'] | None = 'adamw',
            weight_decay: float | None = 1e-5,
            factor: float | None = 0.5,
            penalty: Literal["l1", "l2", "elasticnet"] | None="l2",
            scheduler_patience: int | None = 5,
            early_stopping: bool | None = True,
            l1_ratio: float | None = 0.5,
            random_state: Any | None = None,
            bootstrap: bool | None = True,
            tol: float | None = 1e-4,
            batch_size: int | None = 32
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
        self.delta = huber_threshold

        self.optimizer = optimizer
        self.factor = factor
        self.scheduler_patience = scheduler_patience
        self.shuffle = shuffle
        self.train_patience = train_patience

        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.weight_decay = weight_decay

        self.max_features = max_features
        self.device = device
        self.batch_size = batch_size

        self.loss_history = []
        self.supports = []
        self.feature_indices = []
        self.model_data = []
        self.init_prediction = None

    def _handle_sparse_matrix_indexing(self, X, indices):
        """Handle sparse matrix indexing properly"""
        if issparse(X):
            if hasattr(X, 'tocsr'):
                X = X.tocsr()

            if isinstance(indices, np.ndarray) and len(indices) == X.shape[0]:
                return X[indices].toarray()
            
            else:
                return X[indices]
        else:
            return X[indices]

    def _handle_sparse_matrix_column_selection(self, X, feature_idx):
        if issparse(X):
            return X[:, feature_idx]
        
        else:
            return X[:, feature_idx]

    def fit(self, X: np.ndarray | spmatrix, y: np.ndarray):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if not issparse(X):
            X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Added shape check for robustness
        if len(y) != n_samples:
            raise ValueError(f"Number of samples in X ({n_samples}) and y ({len(y)}) must be equal")

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
        init_pred = np.mean(y)
        self.init_prediction = init_pred  # Store initial prediction
        current_predictions = np.full(n_samples, init_pred, dtype=np.float32)

        for i in range(self.n_estimators):
            residuals = y.astype(np.float32) - current_predictions

            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
            else:
                indices = np.arange(n_samples)

            feature_idx = np.random.choice(n_features, size=max_features, replace=False)
            self.feature_indices.append(feature_idx)

            train_size = int(0.8 * len(indices))
            idx_perm = np.random.permutation(len(indices))
            train_idx = idx_perm[:train_size]
            val_idx = idx_perm[train_size:]

            # Handle sparse matrix indexing properly
            X_sub = self._handle_sparse_matrix_indexing(X, indices)
            y_sub = residuals[indices]

            # Handle column selection
            X_sub = self._handle_sparse_matrix_column_selection(X_sub, feature_idx)

            # Convert to dense if needed for tensor operations
            if issparse(X_sub):
                X_sub = X_sub.toarray()

            X_sub = X_sub.astype(np.float32)

            # Create tensors with proper shapes
            X_train_tensor = torch.tensor(X_sub[train_idx], dtype=torch.float32, device=self.device)
            y_train_tensor = torch.tensor(y_sub[train_idx], dtype=torch.float32, device=self.device)
            X_val_tensor = torch.tensor(X_sub[val_idx], dtype=torch.float32, device=self.device)
            y_val_tensor = torch.tensor(y_sub[val_idx], dtype=torch.float32, device=self.device)

            # Create data loaders
            train_ds = TensorDataset(X_train_tensor, y_train_tensor)
            val_ds = TensorDataset(X_val_tensor, y_val_tensor)

            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=self.shuffle)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            
            if self.model == None:
                support_model = Support(input_dim=max_features)
                support_model.to(self.device)

            else:
                support_model = self.model(input_dim=max_features)
                support_model.to(self.device)

            support = Train(
                train_loader=train_loader,
                val_loader=val_loader,
                model=support_model,
                learning_rate=self.learning_rate,
                loss=self.loss,
                early_stopping=self.early_stop,
                verbose=self.verbose,
                huber_threshold=self.delta,
                optimizer=self.optimizer,
                device=self.device,
                weight_decay=self.weight_decay,
                scheduler_patience=self.scheduler_patience,
                train_patience=self.train_patience,
                epochs=self.max_iter,
                factor=self.factor
            )

            if self.verbose == 2:
                print(f"|=-Training {i + 1}th model-=|")

            model_data = support.train_model()
            self.supports.append(support)
            self.model_data.append(model_data)

            # Get validation loss properly
            if isinstance(model_data, tuple) and len(model_data) >= 2:
                history = model_data[1]
                if 'val_loss' in history and history['val_loss']:
                    final_val_loss = history['val_loss'][-1]
                else:
                    final_val_loss = float('inf')
            else:
                final_val_loss = float('inf')

            self.loss_history.append(final_val_loss)

            # Fixed: Predict on FULL dataset with selected features for update
            X_full_sub = self._handle_sparse_matrix_column_selection(X, feature_idx)
            if issparse(X_full_sub):
                X_full_sub = X_full_sub.toarray()
            X_full_sub = X_full_sub.astype(np.float32)
            X_full_tensor = torch.tensor(X_full_sub, dtype=torch.float32, device=self.device)
            pred_full = support.predict(X_full_tensor)

            if pred_full.ndim > 1:
                pred_full = pred_full.ravel()

            current_predictions += self.learning_rate * pred_full

            if self.verbose == 1:
                print(f"|=Trained {i+1}/{self.n_estimators}th model, Loss: {self.loss_history[-1]:.4f}=|")

            if self.early_stop and i > 2:
                recent_losses = self.loss_history[-3:]
                if max(recent_losses) - min(recent_losses) < self.tol:
                    if self.verbose == 1:
                        print(f"|=Stopping supports training at estimator {i + 1}=|")
                    break

    def predict(self, X: np.ndarray | spmatrix) -> np.ndarray:
        if not issparse(X):
            X = np.asarray(X)

        n_samples = X.shape[0]
        # Fixed: Start with initial prediction
        if self.init_prediction is None:
            raise ValueError("Model has not been fitted yet")
        final_predictions = np.full(n_samples, self.init_prediction, dtype=np.float32)

        for i, support in enumerate(self.supports):
            feature_idx = self.feature_indices[i]

            X_sub = self._handle_sparse_matrix_column_selection(X, feature_idx)

            if issparse(X_sub):
                if X_sub.shape[1] <= 1000:  
                    X_sub = X_sub.toarray()
                else:
                    X_sub = X_sub.toarray()

            X_sub = X_sub.astype(np.float32)
            X_sub_tensor = torch.tensor(X_sub, dtype=torch.float32, device=self.device)

            pred = support.predict(X_sub_tensor)

            if pred.ndim > 1:
                pred = pred.ravel()

            final_predictions += self.learning_rate * pred

        return final_predictions

    def get_params(self, deep=True):
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
            'huber_threshold': self.delta,
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

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self