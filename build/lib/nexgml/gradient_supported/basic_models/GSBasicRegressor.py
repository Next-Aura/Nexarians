# ========== LIBRARIES ==========
import numpy as np                           # Numpy for numerical computations
from scipy.sparse import issparse, spmatrix  # For sparse matrix handling
from typing import Literal, Optional         # More specific type hints
from nexgml.amo import forlinear             # For specific computation operations

# ========== THE MODEL ==========
class BasicRegressor:
    """
    Gradient Supported Basic Regressor (GSBR) is a linear regression model that uses gradient descent optimization to minimize the loss function. 
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
            lr_scheduler: Literal["constant", "invscaling", 'plateau'] | None='invscaling',
            power_t: float=0.25,
            patience: int=5,
            factor: float=0.5,
            stoic_iter: int | None = 10
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

            **lr_scheduler**: *{'constant', 'invscaling', 'plateau'} or None, default='invscaling'*
            Strategy for learning rate adjustment over iterations.

            **power_t**: *float, default=0.25*
            The exponent for inverse scaling learning rate schedule (used if lr_scheduler='invscaling').

            **patience**: *int, default=5*
            Number of epochs to wait for loss improvement before reducing learning rate (used if lr_scheduler='plateau').

            **factor**: *float, default=0.5*
            Factor by which the learning rate will be reduced (used if lr_scheduler='plateau').
            
            **stoic_iter**: *int or None, default=10*
            Number of initial epochs to skip before checking for convergence/tolerance in early stopping.

        ## Returns:
            **None**

        ## Raises:
            **ValueError**: *If invalid penalty, loss, or lr scheduler type is provided.*
        """
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
    def _calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculating loss with regulation, MSE, RMSE and MAE available.
        Penalty, l1, l2, elasticnet available.
        
        ## Args:
            **y_true**: *np.ndarray*
            True target values.

            **y_pred**: *np.ndarray*
            Predicted target values.
            
        ## Returns:
            **float**: *total loss with regulation (if regulation is used).*
            
        ## Raises:
            **None**
        """
        # MSE loss function
        if self.loss == 'mse':
            loss = forlinear.mean_squared_error(y_true, y_pred)
        
        # RMSE loss function
        elif self.loss == 'rmse':
            loss = forlinear.root_squared_error(y_true, y_pred)

        # MAE loss function
        elif self.loss == 'mae':
            loss = forlinear.mean_absolute_error(y_true, y_pred)
        
        # L1 penalty regulation
        if self.penalty == "l1":
          loss += forlinear.lasso(self.weights, self.alpha)
        
        # L2 penalty regulation
        elif self.penalty == "l2":
          loss += forlinear.ridge(self.weights, self.alpha)
        
        # Elastic Net penalty regulation
        elif self.penalty == "elasticnet":
          loss += forlinear.elasticnet(self.weights, self.alpha, self.l1_ratio)
           
        return loss

    def _calculate_grad(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Calculate gradient of loss function with regulation.
        L1, L2, and Elastic Net available.
        
        ## Args:
            **X**: *np.ndarray*
            Input features.

            **y**: *np.ndarray*
            True target values.
            
        ## Return:
            **tuple**: *(np.ndarray, float, np.ndarray).*
            np.ndarray: gradient w.r.t. weights.
            float: gradient w.r.t. bias.
            np.ndarray: Calculated linear combination.
            
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
            grad_w, grad_b = forlinear.mse_deriv(X, error, self.intercept)
        
        # RMSE loss gradient
        elif self.loss == 'rmse':
           grad_w, grad_b = forlinear.rmse_deriv(X, error, self.intercept)
        
        # MAE loss gradient
        elif self.loss == 'mae':
           grad_w, grad_b = forlinear.mae_deriv(X, error, self.intercept)
        
        # L1 penalty gradient
        if self.penalty == "l1":
            grad_w += forlinear.lasso_deriv(self.weights, self.alpha)
        
        # L2 penalty gradient
        elif self.penalty == "l2":
            grad_w += forlinear.ridge_deriv(self.weights, self.alpha)
        
        # Elastic Net penalty gradient
        elif self.penalty == "elasticnet":
            grad_w += forlinear.elasticnet_deriv(self.weights, self.alpha)

        return grad_w, grad_b, f

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
                # Constant learning rate scheduler
                if self.lr_scheduler == 'constant':
                    # Keep learning rate constant
                    self.current_lr = self.current_lr
                
                # Invscaling learning rate scheduler
                elif self.lr_scheduler == 'invscaling':
                    # Inverse scaling decay
                    self.current_lr = self.current_lr / ((i + 1)**self.power_t + self.epsilon)
                
                # Plateau learning rate scheduler
                elif self.lr_scheduler == 'plateau':
                    # Compute full dataset loss
                    current_loss = self._calculate_loss(y_processed, self.weights @ X_processed + self.b if self.intercept else self.weights @ X_processed)
                    if current_loss < self.best_loss - self.epsilon:
                        # Update best loss
                        self.best_loss = current_loss
                        # Reset wait counter
                        self.wait = 0
                    elif abs(current_loss - self.best_loss) < self.tol:
                        # Increment wait counter
                        self.wait += 1
                    else:
                        # Reset wait counter
                        self.wait = 0

                    if self.wait >= self.patience:
                        # Reduce learning rate
                        self.current_lr *= self.factor
                        # Reset wait counter
                        self.wait = 0
                        if self.verbose == 1:
                            print(f"|=-Epoch {i + 1} reducing learning rate to {self.current_lr:.6f}-=|")

            if self.shuffle:
                indices = rng.permutation(num_samples)
                X_processed = X_processed[indices]
                y_processed = y_processed[indices]

            else:
               pass
            
            # Compute gradients
            grad_w, grad_b, pred = self._calculate_grad(X_processed, y_processed)
            
            # Update weight
            self.weights -= self.current_lr * grad_w
            
            # update bias if intercept is used
            if self.intercept:
             self.b -= self.current_lr * grad_b
            
            # Calculate current iteration loss
            loss = self._calculate_loss(y_processed, pred)
            
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
            if self.early_stop and i > 1 and i > self.stoic_iter:
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
    
    def score(self, X_test: np.ndarray | spmatrix, y_test: np.ndarray) -> float:
        """
        Calculate the coefficient of determination R^2 of the prediction.

        ## Args:
            **X_test**: *np.ndarray* or *spmatrix*
            Feature matrix.

            **y_test**: *np.ndarray*
            True target values.

        ## Returns:
            **float**: *R^2 score.*

        ## Raises:
            **None**
        """
        # ========== PREDICTION ==========
        y_pred = self.predict(X_test)
        u = ((y_test - y_pred) ** 2).sum()
        v = ((y_test - y_test.mean()) ** 2).sum()
        return 1 - u / v if v != 0 else 0.0