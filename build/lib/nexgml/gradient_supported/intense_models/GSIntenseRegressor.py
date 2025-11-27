# ========== LIBRARIES =========
import numpy as np                           # For numerical computations 
from scipy.sparse import spmatrix, issparse  # For sparse matrix handling
from typing import Literal, Optional         # More specific type hints
from nexgml.amo import forlinear             # For specific numerical computations
from warnings import warn                    # For warning messages

# ========== THE MODEL ==========
class IntenseRegressor:
    """
    Gradient Supported Intense Regressor (GSIR) is an advanced linear regression model that uses gradient descent optimization with mini-batch support. 
    It supports L1, L2, and Elastic Net regularization to prevent overfitting, multiple optimizers (MBGD, Adam, AdamW), and various learning rate schedulers.
    MSE, RMSE, MAE, and Huber loss functions are available.

    It supports L1, L2, and Elastic Net regularization, along with learning 
    rate scheduling and early stopping to optimize training.
    Handles both dense and sparse input matrices.
    """
    def __init__(
        self, 
        max_iter: int=1000, 
        learning_rate: float=0.01,
        penalty: Optional[Literal["l1", "l2", "elasticnet"]] | None="l2", 
        alpha: float=0.0001, 
        l1_ratio: float=0.5, 
        loss: Literal["mse", "rmse", "mae", "smoothl1"] | None="mse", 
        fit_intercept: bool=True, 
        tol: float=0.0001, 
        shuffle: bool | None=True, 
        random_state: int | None=None, 
        early_stopping: bool=True,
        verbose: int=0,
        verbosity: Literal['light', 'heavy'] | None = 'light',
        lr_scheduler: Literal["constant", "invscaling", "plateau", "adaptive"] | None="invscaling", 
        optimizer: Literal["mbgd", "adam", "adamw"] | None="mbgd", 
        batch_size: int=16, 
        power_t: float=0.25, 
        patience: int=5, 
        factor: float=0.5, 
        delta: int=0.5,
        stoic_iter: int | None = 10
        ):
        """
        Initialize the IntenseRegressor model.
        
        ## Args:
            **max_iter**: *int, default=1000*
            Maximum number of gradient descent iterations (epochs).

            **learning_rate**: *float, default=0.01*
            Initial learning rate (step size) for gradient descent updates.

            **penalty**: *{'l1', 'l2', 'elasticnet'} or None, default='l2'*
            Type of regularization ('l1', 'l2', 'elasticnet') or None.

            **alpha**: *float, default=0.0001*
            Regularization strength (used if penalty is not None).

            **l1_ratio**: *float, default=0.5*
            Mixing parameter for elastic net (0 <= l1_ratio <= 1).

            **loss**: *{'mse', 'rmse', 'mae', 'huber'}, default='mse'*
            Type of loss function. Includes Huber loss for robustness.

            **fit_intercept**: *bool, default=True*
            If True, include a bias term (intercept).

            **tol**: *float, default=0.0001* 
            Tolerance for early stopping based on loss convergence.

            **shuffle**: *bool, default=True*
            If True, shuffle data each epoch.

            **random_state**: *int or None, default=None*
            Seed for random number generator for reproducibility.

            **early_stopping**: *bool, default=True*
            If true, will make the model end the training loop early if the model is in plateau performance 
            or loss convergence is met.

            **verbose**: *int, default=0*
            If 1, print training progress (epoch, loss, weights, bias, LR). If 2, print more detailed LR updates.
            
            **verbosity**: *{'light', 'heavy'}, default='light'*
            Level of detail for verbose output.

            **lr_scheduler**: *{'constant', 'invscaling', 'plateau', 'adaptive'}, default='invscaling'*
            Strategy for learning rate adjustment over iterations.

            **optimizer**: *{'mbgd', 'adam', 'adamw'} or None, default='mbgd'*
            Optimization algorithm to use for gradient descent. 'mbgd' is Mini-Batch Gradient Descent.

            **batch_size**: *int, default=16*
            Number of samples per mini-batch when using mini-batch gradient descent (MBGD, Adam, AdamW).
            
            **power_t**: *float, default=0.25*
            The exponent for inverse scaling learning rate schedule (used if lr_scheduler='invscaling').

            **patience**: *int, default=5*
            Number of epochs to wait for loss improvement before reducing learning rate (used if lr_scheduler='plateau').

            **factor**: *float, default=0.5*
            Factor by which the learning rate will be reduced (used if lr_scheduler='plateau').
            
            **delta**: *float, default=0.5*
            Threshold for the Smooth L1 loss function.

            **stoic_iter**: *int or None, default=10*
            Number of initial epochs to skip before checking for convergence/tolerance in early stopping.

        ## Returns:
            **None**

        ## Raises:
            **ValueError**: *If invalid penalty, loss, optimizer, or lr_scheduler type is provided, 
            or if AdamW is used with a non-L2 penalty.*
            **UserWarning**: *If verbose level 2 is used with heavy verbosity.*
        """

        # ========== PARAMETER VALIDATION ==========
        if penalty not in {"l1", "l2", "elasticnet", None}:
            raise ValueError(f"Invalid penalty argument, {penalty}. Choose from 'l1', 'l2', 'elasticnet', or None")
        
        if verbosity not in ('light', 'heavy'):
            raise ValueError(f"Invalid verbosity argument, {verbosity}. Choose from 'light' or 'heavy'.")

        if loss not in {'mse', 'rmse', 'mae', 'smoothl1'}:
            raise ValueError(f"Invalid loss argument, {loss}. Choose from 'mse', 'rmse', 'mae', or 'huber'")
        
        if optimizer not in {'mbgd', 'adam', 'adamw'}:
            raise ValueError(f"Invalid optimizer argument, {optimizer}. Choose from 'mbgd', 'adam', or 'adamw'")
        
        if lr_scheduler not in {'constant', 'invscaling', 'plateau', 'adaptive'}:
            raise ValueError(f"Invalid lr_scheduler argument, {lr_scheduler}. Choose from 'constant', 'invscaling', 'plateau', or 'adaptive'.")
        
        if penalty in {'l1', 'elasticnet'} and optimizer == 'adamw':
            raise ValueError("AdamW only supports L2 regularization. Please change the penalty to 'l2'")
        
        if verbose == 2 and verbosity == 'heavy':
            warn("Verbose level 2 with heavy verbosity may produce excessive output.", UserWarning)

        # ========== HYPERPARAMETERS ==========
        self.max_iter = int(max_iter)                           # Model max training iterations
        self.learning_rate = float(learning_rate)               # Initial learning rate
        self.penalty = str(penalty)                             # Penalties for regularization
        self.alpha = float(alpha)                               # Alpha for regularization power
        self.l1_ratio = float(l1_ratio)                         # Elastic net mixing ratio
        self.loss = str(loss)                                   # Loss function
        self.intercept = bool(fit_intercept)                    # Fit intercept (bias) or not
        self.tol = float(tol)                                   # Training loss tolerance
        self.shuffle = bool(shuffle)                            # Data shuffling
        self.random_state = random_state                        # Random state for reproducibility
        self.early_stop = bool(early_stopping)                  # Early stopping flag
        self.verbose = int(verbose)                             # Model progress logging
        self.verbosity = str(verbosity)                         # Verbosity level for logging
        
        self.lr_scheduler = str(lr_scheduler)                   # Learning rate scheduler method
        self.optimizer = str(optimizer)                         # Optimizer type
        self.batch_size = int(batch_size)                       # Batch size
        self.power_t = float(power_t)                           # Invscaling power
        self.patience = int(patience)                           # Patience for plateau scheduler
        self.factor = float(factor)                             # Plateau scheduler factor
        self.delta = float(delta)                               # Huber loss threshold
        self.stoic_iter = int(stoic_iter)                       # Warm-up iterations before applying early stopping and lr scheduler
        
        # ========== INTERNAL VARIABLES ==========
        self.epsilon = 1e-15                                    # Small value for stability
        self.loss_history = []                                  # Store loss per-iteration
        self.weights = None                                     # Model weights
        self.b = 0.0                                            # Model bias
        self.current_lr = None                                  # Store current learning rate per-iteration
        self.best_loss = float('inf')                           # Initial best loss for plateau
        self.wait = 0                                           # Wait counter for plateau scheduler
        
        # ---------- Adam/AdamW specific ----------
        if self.optimizer == 'adam' or self.optimizer == 'adamw': 
            self.m_w = None                                     # First moment vector for weights
            self.v_w = None                                     # Second moment vector for weights
            self.beta1 = 0.9                                    # Decay rate for the first moment estimates
            self.beta2 = 0.999                                  # Decay rate for the second moment estimates
            self.m_b = 0.0                                      # First moment for bias
            self.v_b = 0.0                                      # Second moment for bias

    # ========== HELPER METHODS  ==========
    def _calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculating loss with regulation, MSE, RMSE, MAE, and Huber available.
        Penalty, l1, l2, elasticnet available.
        
        ## Args:
            **y_true**: *np.ndarray*
            True target values.

            **y_pred**: *np.ndarray*
            Predicted target values.
            
        ## Returns:
            **float**: *total loss with regulation*
            
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

        # Smooth L1 loss function
        elif self.loss == 'smoothl1':
            loss = forlinear.smoothl1_loss(y_true, y_pred, self.delta)
        
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
        residual = f - y

        # MSE loss gradient
        if self.loss == 'mse':
            grad_w, grad_b = forlinear.mse_deriv(X, residual, self.intercept)
        
        # RMSE loss gradient
        elif self.loss == 'rmse':
           grad_w, grad_b = forlinear.rmse_deriv(X, residual, self.intercept)
        
        # MAE loss gradient
        elif self.loss == 'mae':
           grad_w, grad_b = forlinear.mae_deriv(X, residual, self.intercept)

        # Smooth L1 loss gradient
        elif self.loss == 'smoothl1':
           grad_w, grad_b = forlinear.smoothl1_deriv(X, residual, self.intercept, self.delta)
        
        # L1 penalty gradient
        if self.penalty == "l1":
            grad_w += forlinear.lasso_deriv(self.weights, self.alpha)
        
        # L2 penalty gradient (not for AdamW as it uses weight decay)
        elif self.penalty == "l2" and self.optimizer != "adamw":
            grad_w += forlinear.ridge_deriv(self.weights, self.alpha)

        # L2 penalty gradient for AdamW (zero because weight decay handles it separately)
        elif self.penalty == "l2" and self.optimizer == "adamw":
            grad_w += np.zeros_like(self.weights) 
        
        # Elastic Net penalty gradient
        elif self.penalty == "elasticnet":
            grad_w += forlinear.elasticnet_deriv(self.weights, self.alpha, self.l1_ratio)

        return grad_w, grad_b, f

    # ========== MAIN METHODS (Disesuaikan dengan format GSBR) ==========
    def fit(self, X_train: np.ndarray | spmatrix, y_train: np.ndarray | spmatrix) -> None:
        """
        Fit the model to the training data using gradient descent method (Mini-batch GD, Adam, or AdamW).
        
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
        # Sparse matrix check      
        if isinstance(X_train, spmatrix):
            # Keep as is for sparse (CSR or CSC)
            if X_train.shape[0] > X_train.shape[1]:
              X_processed = X_train.tocsr()

            else:
              X_processed = X_train.tocsc()
        
        # Other data types check
        elif isinstance(X_train, (np.ndarray, list, tuple)):
            # Convert to numpy array
            X_processed = np.asarray(X_train)

        else:
            # Convert DataFrame to numpy array
            X_processed = X_train.to_numpy()
        
        # Check if the data is 1D
        if X_processed.ndim == 1:
            # Reshape to 2D
            X_processed = X_processed.reshape(-1, 1)
        
        # y data type check
        if isinstance(y_train, (np.ndarray, list, tuple)):
            # Convert to numpy array
            y_processed = np.asarray(y_train)

        else:
            # Convert DataFrame to numpy array
            y_processed = y_train.to_numpy()
        
        # Flattening y data
        y_processed = y_processed.ravel()
        
        # Sparse data check
        if issparse(X_processed):
            # Check if sparse data is finite
            if not np.all(np.isfinite(X_processed.data)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")

        else:
            # Check if (another type) data is finite
            if not np.all(np.isfinite(X_processed)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")
        
        # Check if y data is finite
        if not np.all(np.isfinite(y_processed)):
          raise ValueError("Input target (y_train) contains NaN or Infinity values. Please clean your data.")
        
        # Check if X and y has the same samples
        if X_processed.shape[0] != y_processed.shape[0]:
            raise ValueError(
                f"Number of samples in X_train ({X_processed.shape[0]}) "
                f"must match number of samples in y_train ({y_processed.shape[0]})."
            )
        
        # Pre-train process

        # Number of samples and features
        num_samples, num_features = X_processed.shape

        # Set random state
        rng = np.random.default_rng(self.random_state)

        # Number of batches
        num_batch = int(np.ceil(num_samples / self.batch_size))

        # RNG for weight initialization
        rng_init = np.random.default_rng(self.random_state)
        
        # Check if weights is available or matches feature size
        if self.weights is None or self.weights.shape[0] != num_features:
          # Weight initialization
          self.weights = rng_init.normal(loc=0.0, scale=0.01, size=num_features)
      
        if self.intercept:
          # Bias initialization if intercept is used
          self.b = rng_init.normal(loc=0.0, scale=0.01)
        
        # Additional initialization for Adam and AdamW optimizers
        if self.optimizer in ['adam', 'adamw']:
            self.m_w = np.zeros_like(self.weights)
            self.v_w = np.zeros_like(self.weights)
            self.m_b = 0.0
            self.v_b = 0.0
        
        # Current learning rate initialization
        self.current_lr = self.learning_rate

        # ---------- Training loop ----------

        for i in range(self.max_iter):
            if i > self.stoic_iter:
                # Constant learning rate
                if self.lr_scheduler == 'constant':
                    self.current_lr = self.learning_rate
                
                # Invscaling learning rate scheduler
                elif self.lr_scheduler == 'invscaling':
                    self.current_lr = self.current_lr / ((i + 1)**self.power_t + self.epsilon)
                
                elif self.lr_scheduler == 'adaptive':
                    # Adaptive learning rate based on loss ratio
                    self.current_lr = min(10.0, max(self.current_lr * (self.loss_history[-1] / self.loss_history[-2]), 1e-8)) if i > 1 else self.current_lr

                elif self.lr_scheduler == 'plateau':
                        # Get current loss
                        current_loss = self._calculate_loss(y_processed, X_processed @ self.weights + self.b if self.intercept else X_processed @ self.weights)
                        # Check for best loss improvement
                        if current_loss < self.best_loss - self.epsilon:
                            # Update best loss
                            self.best_loss = current_loss
                            # Wait counter
                            self.wait = 0
                        
                        # Check for tolerance
                        elif abs(current_loss - self.best_loss) < self.tol:
                            # Increase the wait counter
                            self.wait += 1

                        else:
                            # Reset wait if loss improves
                            self.wait = 0
                        
                        # Check if patience exceeded
                        if self.wait >= self.patience:
                          # Reduce learning rate
                          self.current_lr *= self.factor
                          # Reset wait counter
                          self.wait = 0
                    
                        if self.verbose == 2 and self.verbosity == 'heavy':
                            # Log learning rate reduce if verbose in level 2 with heavy verbosity
                            print(f"- Epoch {i + 1} reducing learning rate to {self.current_lr:.8f}")
            
            # Shuffle condition
            if self.shuffle:
                # RNG for data shuffle
                indices = rng.permutation(num_samples)
                # Shuffle X data
                X_shuffled = X_processed[indices]
                # Shuffle y data
                y_shuffled = y_processed[indices]
            
            # No shuffle
            else:
                X_shuffled = X_processed
                y_shuffled = y_processed
            
            # Batch processing
            for j in range(num_batch):
                # Start index
                s_idx = j * self.batch_size
                # End index
                e_idx = min((j + 1) * self.batch_size, num_samples)

                # X data slicing
                X_batch = X_shuffled[s_idx:e_idx]
                # y data slicing
                y_batch = y_shuffled[s_idx:e_idx]

                # Check if y is 1D for grad calculation
                grad_w, grad_b, pred = self._calculate_grad(X_batch, y_batch.ravel())
                
                # Mini-batch Gradient Descent optimizer
                if self.optimizer == 'mbgd':
                   # Weight calculation
                   self.weights -= self.current_lr * grad_w
                   
                   # Intercept condition
                   if self.intercept:
                      # Bias calculation
                      self.b -= self.current_lr * grad_b
                
                # Adam optimizer
                elif self.optimizer == 'adam':
                    # Time step
                    t = i * num_batch + j + 1

                    # First moment update for weights
                    self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_w

                    # Second moment update for weights
                    self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_w**2)

                    # Bias-corrected first moment for weights
                    m_w_hat = self.m_w / (1 - self.beta1**t)

                    # Bias-corrected second moment for weights
                    v_w_hat = self.v_w / (1- self.beta2**t)

                    # Weight calculation
                    self.weights -= self.current_lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                    
                    # Intercept condition
                    if self.intercept:
                        # First moment update for bias
                        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b

                        # Second moment update for bias
                        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b**2)

                        # Bias-corrected first moment for bias
                        m_b_hat = self.m_b / (1 - self.beta1**t)

                        # Bias-corrected second moment for bias
                        v_b_hat = self.v_b / (1- self.beta2**t)

                        # Bias calculation
                        self.b -= self.current_lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

                elif self.optimizer == 'adamw':                                                        # AdamW optimizer
                        # Time step
                        t = i * num_batch + j + 1

                        # First moment update for weights
                        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_w

                        # Second moment update for weights
                        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_w**2)

                        # Bias-corrected first moment for weights
                        m_w_hat = self.m_w / (1 - self.beta1**t)

                        # Bias-corrected second moment for weights
                        v_w_hat = self.v_w / (1 - self.beta2**t)

                        # Weight calculation
                        self.weights -= self.current_lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

                        # L2 ragulation for weights decay
                        if self.penalty == 'l2':
                            # L2 ragulation for weights decay
                            self.weights -= self.current_lr * self.alpha * self.weights
                        
                        # Intercept condition
                        if self.intercept:
                            # First moment update for bias
                            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b

                            # Second moment update for bias
                            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b**2)

                            # Bias-corrected first moment for bias
                            m_b_hat = self.m_b / (1 - self.beta1**t)

                            # Bias-corrected second moment for bias
                            v_b_hat = self.v_b / (1 - self.beta2**t)

                            # Bias calculation
                            self.b -= self.current_lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

            # Current loss
            loss = self._calculate_loss(y_batch, pred)

            # Store current loss to loss history
            self.loss_history.append(loss)
            
            # Check of weights or bias is finite during traing
            if not np.all(np.isfinite(self.weights)) or (self.intercept and not np.isfinite(self.b)):
                raise OverflowError(f"Weights or bias became NaN/Inf at epoch {i + 1}. Stopping training early.")
            
            # Check of weights or bias is NaN during training
            if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)) or np.isnan(self.b) or np.isinf(self.b):
                    raise OverflowError(f"There's NaN in epoch {i + 1} during the training process")
            
            # Light verbose logging
            if self.verbose == 1 and ((i % max(1, self.max_iter // 20)) == 0 or i < 5) and self.verbosity == 'light':
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {self.b:.6f}")

            elif self.verbose == 2 and self.verbosity == 'light':
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {self.b:.6f}")

            # Heavy verbose logging
            if self.verbose == 1 and ((i % max(1, self.max_iter // 20)) == 0 or i < 5) and self.verbosity == 'heavy':
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {loss:.8f}, Avg Weights: {np.mean(self.weights):.8f}, Avg Bias: {self.b:.8f}, Current LR: {self.current_lr:.8f}")

            elif self.verbose == 2 and self.verbosity == 'heavy':
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {loss:.8f}, Avg Weights: {np.mean(self.weights):.8f}, Avg Bias: {self.b:.8f}, Current LR: {self.current_lr:.8f}")
                        
            # Early stopping based on tolerance
            if self.early_stop and i > self.stoic_iter and i > 1:
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

    def get_params(self, deep=True) -> dict[str, object]:
        """
        Returns model paramters.

        ## Args:
            **deep**: *bool, default=True*
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        ## Returns:
            **dict**: *Model parameters.*

        ## Raises:
            **None**
        """
        return {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "penalty": self.penalty,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "loss": self.loss,
            "fit_intercept": self.intercept,
            "tol": self.tol,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "early_stopping": self.early_stop,
            "verbose": self.verbose,
            "lr_scheduler": self.lr_scheduler,
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "power_t": self.power_t,
            "patience": self.patience,
            "factor": self.factor,
            "delta": self.delta,
            "stoic_iter": self.stoic_iter
        }

    def set_params(self, **params) -> 'IntenseRegressor':
        """
        Returns model's attribute that ready to set.

        ## Args:
            **params**: *dict*
            Model parameters to set.

        ## Returns:
            **IntenseRegressor**: *The model instance with updated parameters.*

        ## Raises:
            **None**
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self