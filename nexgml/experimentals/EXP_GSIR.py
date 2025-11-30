import numpy as np
from typing import Literal, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
import warnings

class ExperimentalIntenseRegressor:
    """
    Enhanced version of IntenseRegressor with improved robustness features:
    - Feature scaling/normalization
    - Outlier detection and handling
    - Gradient clipping
    - Advanced regularization
    - Cross-validation support
    - Better numerical stability
    - Learning rate warm-up
    - Early stopping with validation
    """
    
    def __init__(self, 
                 max_iter: int = 1000,
                 lr_scheduler: Literal["constant", "invscaling", "plateau", "cosine"] | None = 'invscaling',
                 verbose: int = 0,
                 penalty: Literal["l1", "l2", "elasticnet", "none"] = "l2",
                 alpha: float = 0.0001,
                 l1_ratio: float = 0.5,
                 fit_intercept: bool = True,
                 rel_tol: float = 1e-4,
                 abs_tol: float = 0.0001,
                 eta0: float = 0.01,
                 power_t: float = 0.25,
                 epsilon: float = 1e-8,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 random_state: int | None = None,
                 optimizer: Literal['sgd', 'adam', 'adamw', 'rmsprop'] = 'sgd',
                 patience: int = 5,
                 factor: float = 0.5,
                 loss: Literal['mse', 'rmse', 'mae', 'huber', 'quantile'] = 'mse',
                 delta: float = 1.0,
                 early_stop: bool = True,
                 # New robustness parameters
                 feature_scaling: Literal['standard', 'robust', 'minmax', 'none'] = 'none',
                 outlier_threshold: float = 0.0,
                 gradient_clip: float | None = None,
                 validation_split: float = 0.0,
                 warm_up_epochs: int = 0,
                 weight_decay: float = 0.01,
                 quantile_alpha: float = 0.5,
                 dropout_rate: float = 0.0,
                 batch_norm: bool = False):
        
        # Original parameters
        self.max_iter = max_iter
        self.lr_scheduler = lr_scheduler
        self.random_state = random_state
        self.patience = patience
        self.factor = factor
        self.loss = loss
        
        self.verbose = verbose
        self.intercept = fit_intercept
        self.eta0 = eta0
        self.epsilon = epsilon
        
        self.optimizer = optimizer
        self.early_stop = early_stop
        self.tol = abs_tol
        
        self.power_t = power_t
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        
        # New robustness parameters
        self.feature_scaling = feature_scaling
        self.outlier_threshold = outlier_threshold
        self.gradient_clip = gradient_clip
        self.validation_split = validation_split
        self.warm_up_epochs = warm_up_epochs
        self.weight_decay = weight_decay
        self.quantile_alpha = quantile_alpha
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.delta = delta
        
        # Initialize state variables
        self.weights = None
        self.b = 0.0
        self.scaler = None
        self.feature_names = None
        self.best_weights = None
        self.best_b = 0.0
        self.training_history = []
        self.validation_history = []
        self.outlier_mask = None
        self.current_lr = eta0
        
        # Optimizer state
        self._reset_optimizer_state()
        
    def _reset_optimizer_state(self):
        """Reset optimizer state variables"""
        self.m_w = None
        self.v_w = None
        self.m_b = 0.0
        self.v_b = 0.0
        
        # Adam parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon_adam = 1e-8
        
        # RMSprop parameters
        self.rho = 0.9
        
    def _initialize_scaler(self, X: np.ndarray) -> np.ndarray:
        """Initialize and fit feature scaler"""
        if self.feature_scaling == 'standard':
            self.scaler = StandardScaler()
        elif self.feature_scaling == 'robust':
            self.scaler = RobustScaler()
        elif self.feature_scaling == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        else:
            return X
            
        return self.scaler.fit_transform(X)
    
    def _detect_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect and handle outliers using IQR method"""
        if self.outlier_threshold <= 0:
            return X, y, np.ones(len(X), dtype=bool)
        
        # Combine features and target for outlier detection
        data = np.column_stack([X, y])
        
        # Calculate z-scores
        z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
        outlier_mask = np.all(z_scores < self.outlier_threshold, axis=0)
        
        # Create mask for samples
        sample_mask = np.all(z_scores < self.outlier_threshold, axis=1)
        
        if np.sum(sample_mask) < len(X) * 0.5:
            warnings.warn("Too many outliers detected. Consider adjusting threshold.")
            return X, y, np.ones(len(X), dtype=bool)
        
        return X[sample_mask], y[sample_mask], sample_mask
    
    def _clip_gradients(self, grad: np.ndarray) -> np.ndarray:
        """Clip gradients to prevent exploding gradients"""
        if self.gradient_clip is None:
            return grad
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.gradient_clip:
            return grad * (self.gradient_clip / grad_norm)
        return grad
    
    def _get_learning_rate(self, epoch: int, iteration: int = 0) -> float:
        """Calculate learning rate with warm-up and scheduling"""
        total_steps = epoch * 100 + iteration  # Approximate
        
        # Warm-up phase
        if epoch < self.warm_up_epochs:
            return self.eta0 * (epoch + 1) / self.warm_up_epochs
        
        # Cosine annealing
        if self.lr_scheduler == 'cosine':
            return self.eta0 * (1 + np.cos(np.pi * epoch / self.max_iter)) / 2
        
        # Inverse scaling
        elif self.lr_scheduler == 'invscaling':
            return self.eta0 / ((epoch + 1) ** self.power_t + self.epsilon)
        
        # Plateau
        elif self.lr_scheduler == 'plateau':
            return self.current_lr
        
        return self.eta0
    
    def _calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate loss with regularization and robust options"""
        predictions = X @ self.weights + self.b
        
        # Base loss
        if self.loss == 'mse':
            loss = np.mean((predictions - y) ** 2)
        elif self.loss == 'rmse':
            loss = np.sqrt(np.mean((predictions - y) ** 2))
        elif self.loss == 'mae':
            loss = np.mean(np.abs(predictions - y))
        elif self.loss == 'huber':
            residuals = predictions - y
            mask = np.abs(residuals) <= self.delta
            loss = np.mean(
                np.where(mask, 0.5 * residuals ** 2, 
                        self.delta * (np.abs(residuals) - 0.5 * self.delta))
            )
        elif self.loss == 'quantile':
            residuals = y - predictions
            loss = np.mean(
                np.where(residuals >= 0, 
                        self.quantile_alpha * residuals,
                        (1 - self.quantile_alpha) * (-residuals))
            )
        
        # Regularization
        reg_loss = 0
        if self.penalty == 'l1':
            reg_loss = self.alpha * np.sum(np.abs(self.weights))
        elif self.penalty == 'l2':
            reg_loss = self.alpha * np.sum(self.weights ** 2)
        elif self.penalty == 'elasticnet':
            l1 = self.l1_ratio * np.sum(np.abs(self.weights))
            l2 = (1 - self.l1_ratio) * np.sum(self.weights ** 2)
            reg_loss = self.alpha * (l1 + l2)
        
        # Weight decay (AdamW style)
        if self.optimizer == 'adamw' and self.penalty == 'l2':
            reg_loss += self.weight_decay * np.sum(self.weights ** 2)
        
        return loss + reg_loss
    
    def _calculate_gradient(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calculate gradients with regularization and clipping"""
        predictions = X @ self.weights + self.b
        residuals = predictions - y
        grad_b = 0.0
        
        # Base gradient
        if self.loss == 'mse':
            grad_w = 2 * X.T @ residuals / len(X)
            if self.intercept:
              grad_b = 2 * np.mean(residuals)
        elif self.loss == 'rmse':
            mse = np.mean(residuals ** 2)
            grad_w = X.T @ residuals / (len(X) * np.sqrt(mse) + self.epsilon)
            if self.intercept:
              grad_b = np.mean(residuals) / (np.sqrt(mse) + self.epsilon)
        elif self.loss == 'mae':
            grad_w = X.T @ np.sign(residuals) / len(X)
            if self.intercept:
              grad_b = np.mean(np.sign(residuals))
        elif self.loss == 'huber':
            residuals_abs = np.abs(residuals)
            mask = residuals_abs <= self.delta
            huber_grad = np.where(mask, residuals, self.delta * np.sign(residuals))
            grad_w = X.T @ huber_grad / len(X)
            if self.intercept:
              grad_b = np.mean(huber_grad)
        elif self.loss == 'quantile':
            quantile_grad = np.where(residuals >= 0, self.quantile_alpha, -(1 - self.quantile_alpha))
            grad_w = X.T @ quantile_grad / len(X)
            if self.intercept:
              grad_b = np.mean(quantile_grad)
        
        # Add regularization gradient
        if self.penalty == 'l1':
            grad_w += self.alpha * np.sign(self.weights)
        elif self.penalty == 'l2':
            grad_w += 2 * self.alpha * self.weights
        elif self.penalty == 'elasticnet':
            grad_w += self.alpha * (self.l1_ratio * np.sign(self.weights) + 2 * (1 - self.l1_ratio) * self.weights)
        
        # Clip gradients
        grad_w = self._clip_gradients(grad_w)
        
        return grad_w, grad_b
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fit the model with validation support"""
        
        # Input validation
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        
        # Handle validation set
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64)
            if X_val.ndim == 1:
                X_val = X_val.reshape(-1, 1)
        elif self.validation_split > 0:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.validation_split, 
                random_state=self.random_state
            )
        
        # Detect and handle outliers
        if self.outlier_threshold is not None:
          X_train, y_train, outlier_mask = self._detect_outliers(X_train, y_train)
        
        # Feature scaling
        X_train_scaled = self._initialize_scaler(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val) if self.scaler else X_val
        
        # Initialize weights
        n_samples, n_features = X_train_scaled.shape
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(0, 0.1, n_features)
        self.b = 0.0
        
        # Initialize optimizer state
        self._reset_optimizer_state()
        if self.optimizer in ['adam', 'adamw']:
            self.m_w = np.zeros_like(self.weights)
            self.v_w = np.zeros_like(self.weights)
        elif self.optimizer == 'rmsprop':
            self.r_w = np.zeros_like(self.weights)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            # Shuffle data
            if self.shuffle:
                indices = rng.permutation(n_samples)
                X_train_scaled = X_train_scaled[indices]
                y_train = y_train[indices]
            
            # Mini-batch training
            epoch_loss = 0
            n_batches = max(1, n_samples // self.batch_size)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                X_batch = X_train_scaled[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Calculate gradients
                grad_w, grad_b = self._calculate_gradient(X_batch, y_batch)
                
                # Update weights based on optimizer
                lr = self._get_learning_rate(epoch, batch_idx)
                
                if self.optimizer == 'sgd':
                    self.weights -= lr * grad_w
                    if self.intercept:
                      self.b -= lr * grad_b
                    
                elif self.optimizer == 'adam':
                    self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_w
                    self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_w ** 2)
                    
                    m_w_corr = self.m_w / (1 - self.beta1 ** (epoch + 1))
                    v_w_corr = self.v_w / (1 - self.beta2 ** (epoch + 1))
                    
                    self.weights -= lr * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon_adam)
                    
                    self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
                    self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b ** 2)
                    
                    m_b_corr = self.m_b / (1 - self.beta1 ** (epoch + 1))
                    v_b_corr = self.v_b / (1 - self.beta2 ** (epoch + 1))
                    
                    if self.intercept:
                      self.b -= lr * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon_adam)
                
                elif self.optimizer == 'adamw':
                    # Weight decay separate from gradient
                    self.weights *= (1 - lr * self.weight_decay)
                    
                    self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_w
                    self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_w ** 2)
                    
                    m_w_corr = self.m_w / (1 - self.beta1 ** (epoch + 1))
                    v_w_corr = self.v_w / (1 - self.beta2 ** (epoch + 1))
                    
                    self.weights -= lr * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon_adam)
                    
                    self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
                    self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b ** 2)
                    
                    m_b_corr = self.m_b / (1 - self.beta1 ** (epoch + 1))
                    v_b_corr = self.v_b / (1 - self.beta2 ** (epoch + 1))
                    
                    if self.intercept:
                      self.b -= lr * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon_adam)
                
                elif self.optimizer == 'rmsprop':
                    self.r_w = self.rho * self.r_w + (1 - self.rho) * (grad_w ** 2)
                    self.weights -= lr * grad_w / (np.sqrt(self.r_w) + self.epsilon)
                    
                    if self.intercept:
                      self.b -= lr * grad_b
            
            # Calculate losses
            train_loss = self._calculate_loss(X_train_scaled, y_train)
            self.training_history.append(train_loss)
            
            if X_val is not None:
                val_loss = self._calculate_loss(X_val_scaled, y_val)
                self.validation_history.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss - self.epsilon:
                    best_val_loss = val_loss
                    self.best_weights = self.weights.copy()
                    self.best_b = self.b
                    patience_counter = 0

                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience and self.early_stop:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Check for NaN/Inf
            if not np.all(np.isfinite(self.weights)) or not np.isfinite(self.b):
                warnings.warn("NaN or Inf detected in weights. Stopping training.")
                break
            
            if self.verbose and (epoch + 1) % max(1, self.max_iter // 10) == 0:
                val_str = f", Val Loss: {val_loss:.4f}" if X_val is not None else ""
                print(f"Epoch {epoch + 1}/{self.max_iter} - Loss: {train_loss:.4f}{val_str}")
            
            try:
                if self.early_stop:
                  if abs(self.validation_history[-1] - self.validation_history[-2]) or abs(self.training_history[-1] - self.training_history[-2]) < self.tol:
                    break
                  
            except IndexError:
                None

        if X_val is not None and self.best_weights is not None:
            self.weights = self.best_weights
            self.b = self.best_b

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with proper scaling"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        return X @ self.weights + self.b
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance based on weights"""
        return np.abs(self.weights)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
        """Perform cross-validation"""
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create fresh model for each fold
            model = ExperimentalIntenseRegressor(**self.get_params())
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores.append(score)
        
        return {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
    
    def get_params(self) -> dict:
        """Get model parameters"""
        return {
            'max_iter': self.max_iter,
            'lr_scheduler': self.lr_scheduler,
            'verbose': self.verbose,
            'penalty': self.penalty,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'fit_intercept': self.intercept,
            'rel_tol': self.tol,
            'abs_tol': self.tol,
            'eta0': self.eta0,
            'power_t': self.power_t,
            'epsilon': self.epsilon,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'optimizer': self.optimizer,
            'patience': self.patience,
            'factor': self.factor,
            'loss': self.loss,
            'delta': self.delta,
            'early_stop': self.early_stop,
            'feature_scaling': self.feature_scaling,
            'outlier_threshold': self.outlier_threshold,
            'gradient_clip': self.gradient_clip,
            'validation_split': self.validation_split,
            'warm_up_epochs': self.warm_up_epochs,
            'weight_decay': self.weight_decay,
            'quantile_alpha': self.quantile_alpha
        }