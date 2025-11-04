# ========== LIBRARIES ==========
import numpy as np                           # Numpy for numerical computations
from scipy.sparse import issparse, spmatrix  # For sparse matrix handling
from typing import Literal, Optional         # More specific type hints
from nexgml.helper.amo import AMO            # For some math operation

# ========== THE MODEL ==========
class BasicRegressor:
    """
    Gradient Supported Basic Regressor (GSBR) is a simple linear regression model that uses gradient descent optimization to minimize the loss function. 
    It supports L1, L2, and Elastic Net regularization to prevent overfitting.
    """
    def __init__(
            self, 
            max_iter: int=1000, 
            learning_rate: float=0.01, 
            penalty: Optional[Literal["l1", "l2", "elasticnet"]] | None="l2", 
            alpha: float=0.0001, 
            l1_ratio: float=0.5, 
            loss: Literal["mse", "rmse", 'mae'] | None="mse",
            fit_intercept: bool=True, 
            tol: float=0.0001,
            shuffle: bool | None=True,
            random_state: int | None=None,
            early_stopping: bool=True,
            verbose: int=0,
            ):
        """
        Initialize the BasicRegressor model.
        
        ## Args:
            **max_iter**: *int, default=1000*
            Maximum number of gradient descent iterations.

            **learning_rate**: *float, default=0.01*
            Step size for gradient descent updates.

            **penalty**: *{'l1', 'l2', 'elasticnet'} or None, default='l2'*
            Type of regularization ('l1', 'l2', 'elasticnet') or None.

            **alpha**: *float, default=0.0001*
            Regularization strength (used if penalty is not None).

            **l1_ratio**: *float, default=0.5*
            Mixing parameter for elastic net (0 <= l1_ratio <= 1).

            **loss**: *{'mse', 'rmse', 'mae'}, default='mse'*
            Type of loss function.

            **fit_intercept**: *bool, default=True*
            If True, include a bias term (intercept).

            **tol**: *float, default=0.0001* 
            Tolerance for early stopping based on loss convergence.

            **shuffle**: *bool, default=True*
            If True, shuffle data each epoch.

            **random_state**: *float, default=None*
            Seed for random number generator for reproducibility.

            **early_stopping**: *bool, default=True*
            If true, will make the model end the training loop early if the model in plateau performance.

            **verbose**: *int, default=0*
            If 1, print training progress (epoch, loss, etc.).

        ## Returns:
            **None**

        ## Raises:
            **ValueError**: *If invalid penalty or loss type is provided.*
        """
        # ========== PARAMETER VALIDATIONS ==========
        if penalty not in (None, "l1", "l2", "elasticnet"):
           raise ValueError(f"Invalid penalty argument {penalty}")

        if loss not in ('mse', 'rmse', 'mae'):
            raise ValueError(f"Invalid loss argument {loss}")
        
        # ========== HYPERPARAMETERS ==========
        self.max_iter = int(max_iter)              # Model max training iterations
        self.learning_rate = float(learning_rate)  # Learning rate for gradient descent
        self.verbose = int(verbose)                # Model progress logging
        self.intercept = bool(fit_intercept)       # Fit intercept (bias) or not
        self.random_state = random_state           # Random state for reproducibility

        self.tol = float(tol)                      # Training loss tolerance for early stopping
        self.shuffle = shuffle                     # Data shuffling
        self.loss = loss                           # Loss function
        self.early_stop = bool(early_stopping)     # Early stopping flag

        self.penalty = penalty                     # Penalties for regularization
        self.l1_ratio = float(l1_ratio)            # Elastic net mixing ratio
        self.alpha = float(alpha)                  # Alpha for regularization power

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
            loss = AMO.mean_squared_error(y, f)
        
        # RMSE loss function
        elif self.loss == 'rmse':
            loss = AMO.root_squared_error(y, f)

        # MAE loss function
        elif self.loss == 'mae':
            loss = AMO.mean_absolute_error(y, f)

        penalty = 0             # Penalty initialization
        
        # L1 penalty regulation
        if self.penalty == "l1":
          penalty = self.alpha * np.sum(np.abs(self.weights))
        
        # L2 penalty regulation
        elif self.penalty == "l2":
          penalty = self.alpha * np.sum(self.weights**2)
        
        # Elastic Net penalty regulation
        elif self.penalty == "elasticnet":
          # L1 part
          l1 = self.l1_ratio * np.sum(np.abs(self.weights))
          # L2 part
          l2 = (1 - self.l1_ratio) * np.sum(self.weights**2)
          # Total with alpha as regulation strength
          penalty = self.alpha * (l1 + l2)
           
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

        grad_b = 0.0         # Initialize bias
        
        # MSE loss gradient
        if self.loss == 'mse':
            grad_w = X.T @ (2 * error) / len(X)
            
            # Calculate bias gradient if intercept is used
            if self.intercept:
              grad_b = np.mean(2 * error)
        
        # RMSE loss gradient
        elif self.loss == 'rmse':
           rmse = np.sqrt(np.mean(error**2))
           grad_w = (X.T @ (2 * error)) / (X.shape[0] * rmse + 1e-10)
           
           # Calculate bias gradient if intercept is used
           if self.intercept:
              grad_b = np.mean(2 * error) / (rmse + 1e-10)
        
        # MAE loss gradient
        elif self.loss == 'mae':
           grad_w = X.T @ np.sign(error) / len(X)
           
           # Calculate bias gradient if intercept is used
           if self.intercept:
            grad_b = np.mean(np.sign(error))

        grad_w_penalty = np.zeros_like(self.weights)    # Initialize gradient penalty
        
        # L1 penalty gradient
        if self.penalty == "l1":
            grad_w_penalty = self.alpha * np.sign(self.weights)
        
        # L2 penalty gradient
        elif self.penalty == "l2":
            grad_w_penalty = 2 * self.alpha * self.weights
        
        # Elastic Net penalty gradient
        elif self.penalty == "elasticnet":
            # L1 part
            l1 = self.l1_ratio * np.sign(self.weights)
            # L2 part
            l2 = 2 * ((1 - self.l1_ratio) * self.weights)
            # Total with alpha as regulation strength
            grad_w_penalty = self.alpha * (l2 + l1)
        
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
        
        # ---------- Training loop ----------
        for i in range(self.max_iter):
            if self.shuffle:
                indices = rng.permutation(num_samples)
                X_processed = X_processed[indices]
                y_processed = y_processed[indices]

            else:
               pass
            
            # Compute gradients
            grad_w, grad_b = self._calculate_grad(X_processed, y_processed)
            
            # Update weight
            self.weights -= self.learning_rate * grad_w
            
            # update bias if intercept is used
            if self.intercept:
             self.b -= self.learning_rate * grad_b
            
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