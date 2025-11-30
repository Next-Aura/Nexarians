# ========== LIBRARIES ==========
import numpy as np                           # Numpy for numerical computations
from scipy.sparse import issparse, spmatrix  # For sparse matrix handling
from typing import Literal, Optional         # More specific type hints
from nexgml import amo                       # For specific computation operations

# ========== THE MODEL ==========
class ImpurityRegressor:
    def __init__(
            self, 
            max_iter: int=1000, 
            learning_rate: float=0.01, 
            penalty: Optional[Literal["l1", "l2", "elasticnet"]] | None="l2", 
            alpha: float=0.0001, 
            l1_ratio: float=0.5, 
            loss: Literal["mse", 'mae'] | None="mse",
            fit_intercept: bool=True, 
            tol: float=0.0001,
            shuffle: bool | None=True,
            random_state: int | None=None,
            early_stopping: bool=True,
            verbose: int=0,
            lr_scheduler: Literal["constant", "invscaling", 'plateau'] | None='invscaling',
            power_t: float=0.25,
            patience: int=5,
            factor: float=0.5,
            stoic_iter: int | None = 10
            ):
        # ========== PARAMETER VALIDATIONS ==========
        if penalty not in (None, "l1", "l2", "elasticnet"):
           raise ValueError(f"Invalid penalty argument, {penalty}. Choose from 'l1', 'l2', or 'elasticnet'.")

        if loss not in ('mse', 'rmse', 'mae'):
            raise ValueError(f"Invalid loss argument, {loss}. Choose from 'mse', 'rmse', or 'mae'.")
        
        if lr_scheduler not in {'invscaling', 'constant', 'plateau'}:
            raise ValueError(f"Invalid lr_scheduler argument {lr_scheduler}. Choose from 'invscaling', 'constant', or 'plateau'.")


        # ========== HYPERPARAMETERS ==========
        self.max_iter = int(max_iter)              # Model max training iterations
        self.learning_rate = float(learning_rate)  # Learning rate for gradient descent
        self.penalty = penalty                     # Penalties for regularization
        self.verbose = int(verbose)                # Model progress logging
        self.intercept = bool(fit_intercept)       # Fit intercept (bias) or not
        self.random_state = random_state           # Random state for reproducibility

        self.tol = float(tol)                      # Training loss tolerance for early stopping
        self.shuffle = shuffle                     # Data shuffling
        self.loss = loss                           # Loss function
        self.early_stop = bool(early_stopping)     # Early stopping flag
        self.lr_scheduler = lr_scheduler           # Learning rate scheduler type ('invscaling', 'constant', 'plateau')
        self.power_t = float(power_t)              # Power parameter for inverse scaling learning rate scheduler
        self.patience = int(patience)              # Number of epochs to wait before reducing learning rate (plateau)
        self.factor = float(factor)                # Factor by which to reduce learning rate on plateau
        self.stoic_iter = int(stoic_iter)          # Warm-up iterations before applying early stopping and lr scheduler

        self.l1_ratio = float(l1_ratio)            # Elastic net mixing ratio
        self.alpha = float(alpha)                  # Alpha for regularization power
        self.current_lr = None                     # Current epoch learning rate
        self.best_loss = float('inf')              # Best loss achieved (used for plateau scheduler)
        self.wait = 0                              # Counter for epochs without improvement (plateau scheduler)
        self.epsilon = 1e-15                       # For numerical stability

        self.loss_history = []                     # Store loss per-iteration
        self.weights = None                        # Moddel weight
        self.b = 0.0                               # Model bias

    # ========== HELPER METHODS ==========
    def _calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating loss with regulation, MSE, RMSE and MAE available.
        Penalty, l1, l2, elasticnet available.
        
        ## Args:
            **X**: *np.ndarray*
            Input features.

            **y**: *np.ndarray*
            True target values.
            
        ## Returns:
            **float**: *total loss with regulation*
            
        ## Raises:
            **None**
        """
        # Linear combination
        f = X @ self.weights
        
        # Add bias if intercept is used
        if self.intercept:
           f += self.b
        
        # MSE loss function
        if self.loss == 'mse':
            loss = amo.fortree.squared_error(y)

        # MAE loss function
        elif self.loss == 'mae':
            loss = amo.fortree.absolute_error(y)

        penalty = 0             # Penalty initialization
        
        # L1 penalty regulation
        if self.penalty == "l1":
          penalty = amo.lasso(self.weights, self.alpha)
        
        # L2 penalty regulation
        elif self.penalty == "l2":
          penalty = amo.ridge(self.weights, self.alpha)
        
        # Elastic Net penalty regulation
        elif self.penalty == "elasticnet":
          penalty = amo.elasticnet(self.weights, self.alpha, self.l1_ratio)
           
        return loss + penalty

    def _calculate_grad(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Calculate gradient of loss function with regulation.
        L1, L2, and Elastic Net available.
        
        ## Args:
            **X**: *np.ndarray*
            Input features.

            **y**: *np.ndarray*
            True target values.
            
        ## Return:
            **tuple**: *gradient w.r.t. weights, gradient w.r.t. bias*
            
        ## Raises:
            **None**
        """
        # Linear combination
        f = X @ self.weights
        
        # Add bias if intercept is used
        if self.intercept:
           f += self.b
        
        # Calculate error (residual)
        error = f - y
        
        # MSE loss gradient
        if self.loss == 'mse':
            grad_w, grad_b = amo.mse_deriv(X, error, self.intercept)
        
        # RMSE loss gradient
        elif self.loss == 'rmse':
           grad_w, grad_b = amo.rmse_deriv(X, error, self.intercept)
        
        # MAE loss gradient
        elif self.loss == 'mae':
           grad_w, grad_b = amo.mae_deriv(X, error, self.intercept)

        grad_w_penalty = np.zeros_like(self.weights)    # Initialize gradient penalty
        
        # L1 penalty gradient
        if self.penalty == "l1":
            grad_w_penalty = amo.lasso_deriv(self.weights, self.alpha)
        
        # L2 penalty gradient
        elif self.penalty == "l2":
            grad_w_penalty = amo.ridge_deriv(self.weights, self.alpha)
        
        # Elastic Net penalty gradient
        elif self.penalty == "elasticnet":
            grad_w_penalty = amo.elasticnet_deriv(self.weights, self.alpha)
        
        # Sum weight gradient with penalty
        grad_w = grad_w + grad_w_penalty

        return grad_w, grad_b

    # ========== MAIN METHODS ==========
    def fit(self, X_train: np.ndarray | spmatrix, y_train: np.ndarray | spmatrix) -> None:
        """
        Fit the model to the training data using gradient descent method.
        
        ## Args:
            **X_train**: *np.ndarray* or *spmatrix*
            Training input features.

            **y_train**: *np.ndarray* or *spmatrix*
            Training target values.
            
        ## Returns:
            **None**
            
        ## Raises:
            **ValueError**: *If input data contains NaN/Inf or if dimensions mismatch.*
            **OverflowError**: *If parameters (weight, bias or loss) become infinity or NaN during training loop.*
            """
        # Check if non-sparse data is 1D and reshape to 2D if is it
        if not issparse(X_train):
          if X_train.ndim == 1:
            X_processed = X_train.reshape(-1, 1)

          else:
            X_processed = np.asarray(X_train)

        # Keep sparse (CSR or CSC)
        else:
            if X_train.shape[0] > X_train.shape[1]:
              X_processed = X_train.tocsr()

            else:
              X_processed = X_train.tocsc()
        
        # Get data shape
        num_samples, num_features = X_processed.shape
        
        # Weight initialize if weight is None or the shape is mismatch with data
        if self.weights is None or self.weights.shape[0] != num_features:
         self.weights = np.zeros(num_features)
        
        # Make sure y is an array data
        if isinstance(y_train, (np.ndarray, list, tuple)):
            y_processed = np.asarray(y_train)
        
        else:
            y_processed = y_train.to_numpy()
        
        # Flattening y data
        y_processed = y_processed.ravel()
        
        # Random state setup for shuffling
        rng = np.random.default_rng(self.random_state)
        
        # Check if there's a NaN in X data (handle sparse and dense separately)
        if issparse(X_processed):
            if not np.all(np.isfinite(X_processed.data)):
                raise ValueError("X_train contains NaN or Infinity values in its data.")
        else:
            if not np.all(np.isfinite(X_processed)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")
        
        # Check if there's a NaN in y data
        if not np.all(np.isfinite(y_processed)):
          raise ValueError("Input target (y_train) contains NaN or Infinity values. Please clean your data.")
        
        # Check if X and y data has the same sample shape
        if X_processed.shape[0] != y_processed.shape[0]:
            raise ValueError(
                f"Number of samples in X_train ({X_processed.shape[0]}) "
                f"must match number of samples in y_train ({y_processed.shape[0]})."
            )
        
        # Current learning rate initialization
        self.current_lr = self.learning_rate
        
        # ---------- Training loop ----------
        for i in range(self.max_iter):
            # Apply LR scheduler after warm-up iterations
            if i > self.stoic_iter:
                if self.lr_scheduler == 'constant':
                    self.current_lr = self.current_lr
                    # Keep learning rate constant

                elif self.lr_scheduler == 'invscaling':
                    self.current_lr = self.current_lr / ((i + 1)**self.power_t + self.epsilon)
                    # Inverse scaling decay

                elif self.lr_scheduler == 'plateau':
                    current_loss = self._calculate_loss(X_processed, y_processed)
                    # Compute full dataset loss
                    if current_loss < self.best_loss - self.epsilon:
                        self.best_loss = current_loss
                        # Update best loss
                        self.wait = 0
                        # Reset wait counter
                    elif abs(current_loss - self.best_loss) < self.tol:
                        self.wait += 1
                        # Increment wait counter
                    else:
                        self.wait = 0
                        # Reset wait counter

                    if self.wait >= self.patience:
                        self.current_lr *= self.factor
                        # Reduce learning rate
                        self.wait = 0
                        # Reset wait counter
                        if self.verbose == 1:
                            print(f"|=-Epoch {i + 1} reducing learning rate to {self.current_lr:.6f}-=|")

            if self.shuffle:
                indices = rng.permutation(num_samples)
                X_processed = X_processed[indices]
                y_processed = y_processed[indices]

            else:
               pass
            
            # Compute gradients
            grad_w, grad_b = self._calculate_grad(X_processed, y_processed)
            
            # Update weight
            self.weights -= self.current_lr * grad_w
            
            # update bias if intercept is used
            if self.intercept:
             self.b -= self.current_lr * grad_b
            
            # Calculate current iteration loss
            loss = self._calculate_loss(X_processed, y_processed)
            
            # Store the calculated curent iteration loss
            self.loss_history.append(loss)
            
            # Check if weight and bias not become an infinite during training loop
            if not np.all(np.isfinite(self.weights)) or (self.intercept and not np.isfinite(self.b)):
                raise OverflowError(f"Weights or bias became NaN/Inf at epoch {i + 1}. Stopping training early.")
            
            # Check if weight and bias not become a NaN and infinity during training loop
            if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)) or np.isnan(self.b) or np.isinf(self.b):
                raise OverflowError(f"There's NaN in epoch {i + 1} during the training process")
            
            # Verbose for training loop logging
            if self.verbose == 1 and ((i % max(1, self.max_iter // 20)) == 0 or i < 5):
                print(f"Epoch {i+1}/{self.max_iter}. Loss: {loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {self.b:.6f}")

            elif self.verbose == 2:
                print(f"Epoch {i+1}/{self.max_iter}. Loss: {loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {self.b:.6f}")
            
            # Early stopping based on loss convergence
            if self.early_stop and i > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                  break

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given input features using the trained model.

        ## Args:
            **X_test**: *np.ndarray*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *predicted target values*

        ## Raises:
            **ValueError**: *If model weights are not defined (model not trained).*
        """
        # Check if data is 1D and reshape to 2D if is it
        if X_test.ndim == 1:
            X_processed = X_test.reshape(-1, 1)
        
        # Or let it as is
        else:
            X_processed = X_test
        
        # Raise an error if weight is None
        if self.weights is None:
            raise ValueError("Weight not defined, try to train the model with fit() function first")
        
        # Linear combination for the prediction
        pred = X_processed @ self.weights + self.b

        return pred