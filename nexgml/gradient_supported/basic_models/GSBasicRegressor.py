# ========== LIBRARIES ==========
import numpy as np                           # Numpy for numerical computations
from scipy.sparse import issparse, spmatrix  # For sparse matrix handling
from typing import Literal, Optional         # More specific type hints
from nexgml.amo import forlinear             # For specific computation operations
from warnings import warn                    # For warning messages
from nexgml.metrics import r2_score          # For R2 score calculation
from nexgml.guardians import safe_array      # For numerical stability

# ========== THE MODEL ==========
class BasicRegressor:
    """
    Gradient Supported Basic Regressor (GSBR) is a linear regression model that uses gradient descent optimization to minimize the loss function. 
    It supports L1, L2, and Elastic Net regularization to prevent overfitting.

    ## Attrs:
      **weights**: *np.ndarray*
      An array that stored features weight with shape (n_feature,).

      **b**: *float*
      A float that is a bias model bias for flexibility output.

      **loss_history**: list[float,...]
      A list that stored loss from all iteration, loss_history is plot-able.

      **current_lr**: *None* (when model not trained, if model is trained the type is 'float')
      Current iteration learning rate.

      **best_loss**: *float*
      Store best loss from model training progress, used for 'plateau' lr_scheduler mechanism.

      **wait**: *int*
      Patience counter for 'plateau' lr_scheduler.

    ## Methods:
      **_calculate_loss(y_true, y_pred)**: *Return float*
      Calculate model loss during training loop.

      **_calculate_grad(X, y)**: *Returns np.ndarray, float, np.ndarray*
      Calculate gradient of selected loss function, 
      return weight and bias gradient and also the linear combination.

      **fit(X_train, y_train)**: *Return None*
      Train model with inputed X_train and y_train argument data.

      **predict(X_test)**: *Return np.ndarray*
      Predict using weights from training session.

      **score(X_test)**: *Return float*
      Calculate model classification accuracy.

      **get_params(deep)**: *Return dict*
      Return model's parameter.

      **set_params([params])**: *Return model's class*
      Set model parameter.

    ## Notes:
      Model is fully implemented on python that may be easy to understand for beginners,
      but also may cause a big latency comparing to another libraries models.

    ## Usage Example:
    ```python
      >>> model = BasicRegressor(loss='mae')
      >>> model.fit(X_train, y_train)
      >>>
      >>> acc = model.score(X_test)
      >>> print("BasicRegressor accuracy:", acc)
    ```
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
            verbosity: Literal['light', 'heavy'] | None = 'light',
            lr_scheduler: Literal["constant", "invscaling", 'plateau', 'adaptive'] | None='invscaling',
            power_t: float=0.25,
            patience: int=5,
            factor: float=0.5,
            stoic_iter: int | None = 10,
            epsilon: float=1e-15
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

            **verbosity**: *{'light', 'heavy'}, default='light'*
            Level of detail for verbose output.

            **lr_scheduler**: *{'constant', 'invscaling', 'plateau', 'adaptive'}, default='invscaling'*
            Strategy for learning rate adjustment over iterations.

            **power_t**: *float, default=0.25*
            The exponent for inverse scaling learning rate schedule (used if lr_scheduler='invscaling').

            **patience**: *int, default=5*
            Number of epochs to wait for loss improvement before reducing learning rate (used if lr_scheduler='plateau').

            **factor**: *float, default=0.5*
            Factor by which the learning rate will be reduced (used if lr_scheduler='plateau').
            
            **stoic_iter**: *int or None, default=10*
            Number of initial epochs to skip before checking for convergence/tolerance in early stopping.
            
            **epsilon**: *float, default=1e-15*
            Small value to avoid numerical instability in calculations.

        ## Returns:
            **None**

        ## Raises:
            **ValueError**: *If invalid penalty, loss, verbosity, or lr scheduler type is provided.*
            **UserWarning**: *If verbose level 2 is used with heavy verbosity.*
        """
        # ========== PARAMETER VALIDATIONS ==========
        if penalty not in (None, "l1", "l2", "elasticnet"):
           raise ValueError(f"Invalid penalty argument, {penalty}. Choose from 'l1', 'l2', or 'elasticnet'.")

        if loss not in ('mse', 'rmse', 'mae'):
            raise ValueError(f"Invalid loss argument, {loss}. Choose from 'mse', 'rmse', or 'mae'.")
        
        if lr_scheduler not in {'invscaling', 'constant', 'plateau', 'adaptive'}:
            raise ValueError(f"Invalid lr_scheduler argument, {lr_scheduler}. Choose from 'invscaling', 'constant', 'plateau', or 'adaptive'.")
        
        if verbosity not in ('light', 'heavy'):
            raise ValueError(f"Invalid verbosity argument, {verbosity}. Choose from 'light' or 'heavy'.")
        
        if verbose == 2 and verbosity == 'heavy':
            warn("Verbose level 2 with heavy verbosity may produce excessive output.", UserWarning)

        # ========== HYPERPARAMETERS ==========
        self.max_iter = int(max_iter)              # Model max training iterations
        self.learning_rate = float(learning_rate)  # Learning rate for gradient descent
        self.penalty = penalty                     # Penalties for regularization
        self.verbose = int(verbose)                # Model progress logging
        self.verbosity = str(verbosity)            # Verbosity level for logging
        self.intercept = bool(fit_intercept)       # Fit intercept (bias) or not
        self.random_state = random_state           # Random state for reproducibility
        self.epsilon = float(epsilon)              # For numerical stability

        self.tol = float(tol)                      # Training loss tolerance for early stopping
        self.shuffle = bool(shuffle)               # Data shuffling
        self.loss = str(loss)                      # Loss function
        self.early_stop = bool(early_stopping)     # Early stopping flag
        self.lr_scheduler = lr_scheduler           # Learning rate scheduler type ('invscaling', 'constant', 'plateau', 'adaptive')
        self.power_t = float(power_t)              # Power parameter for inverse scaling learning rate scheduler
        self.patience = int(patience)              # Number of epochs to wait before reducing learning rate (plateau)
        self.factor = float(factor)                # Factor by which to reduce learning rate on plateau
        self.stoic_iter = int(stoic_iter)          # Warm-up iterations before applying early stopping and lr scheduler

        self.l1_ratio = float(l1_ratio)            # Elastic net mixing ratio
        self.alpha = float(alpha)                  # Alpha for regularization power
        self.current_lr = None                     # Current epoch learning rate
        self.best_loss = float('inf')              # Best loss achieved (used for plateau scheduler)
        self.wait = 0                              # Counter for epochs without improvement (plateau scheduler)

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
            **OverflowError**: *If parameters (weight, bias, loss) become infinity during training loop.*
            **RuntimeWarning**: *If overflow is detected during training process.*
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

                elif self.lr_scheduler == 'adaptive':
                    # Adaptive learning rate based on loss ratio
                    self.current_lr = min(10.0, max(self.current_lr * (self.loss_history[-1] / self.loss_history[-2]), 1e-8)) if i > 1 else self.current_lr
                
                # Plateau learning rate scheduler
                elif self.lr_scheduler == 'plateau':
                    # Compute full dataset loss
                    current_loss = self._calculate_loss(y_processed, X_processed @ self.weights + self.b if self.intercept else X_processed @ self.weights)
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
                        if self.verbose == 2 and self.verbosity == 'heavy':
                            print(f"- Epoch {i + 1} reducing learning rate to {self.current_lr:.8f}.")

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

            if np.isnan(loss):
                loss = safe_array(loss)
            
            # Store the calculated curent iteration loss
            self.loss_history.append(loss)

            # Check if weight and bias not become a NaN and infinity during training loop
            if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)) or np.isnan(self.b) or np.isinf(self.b):
                self.weights = safe_array(self.weights)

                if self.intercept:
                    self.b = safe_array(self.b)
            
            # Check if weight and bias not become an infinite during training loop
            if not np.all(np.isfinite(self.weights)) or (self.intercept and not np.isfinite(self.b)):
                raise OverflowError(f"Weights or bias became NaN/Inf at epoch {i + 1}. Stopping training early.")

            # Check loss for NaN/Inf during training loop
            if not np.isfinite(loss):
                raise OverflowError(f"Loss became NaN/Inf at epoch {i + 1}. Stopping training early.")
            
            # Verbose with light verbosity for training loop logging
            if self.verbose == 1 and ((i % max(1, self.max_iter // 20)) == 0 or i < 5) and self.verbosity == 'light':
                print(f"Epoch {i+1}/{self.max_iter}. Loss: {loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {self.b:.6f}")

            elif self.verbose == 2 and self.verbosity == 'light':
                print(f"Epoch {i+1}/{self.max_iter}. Loss: {loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {self.b:.6f}")

            # Verbose with heavy verbosity for training loop logging
            if self.verbose == 1 and ((i % max(1, self.max_iter // 20)) == 0 or i < 5) and self.verbosity == 'heavy':
                print(f"Epoch {i+1}/{self.max_iter}. Loss: {loss:.8f}, Avg Weights: {np.mean(self.weights):.8f}, Avg Bias: {self.b:.8f}, Current LR: {self.current_lr:.8f}")

            elif self.verbose == 2 and self.verbosity == 'heavy':
                print(f"Epoch {i+1}/{self.max_iter}. Loss: {loss:.8f}, Avg Weights: {np.mean(self.weights):.8f}, Avg Bias: {self.b:.8f}, Current LR: {self.current_lr:.8f}")
            
            # ========== EARLY STOPPING ==========
            if self.early_stop and i > self.stoic_iter:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                    break 
                
                if i > 2 * self.stoic_iter:
                    if abs(np.mean(self.loss_history[-self.stoic_iter:]) - np.mean(self.loss_history[-2*self.stoic_iter:-self.stoic_iter])) < self.tol:
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
        
        # ========== R2 SCORE CALCULATION ==========
        return r2_score(y_test, y_pred)

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
            "verbosity": self.verbosity,
            "lr_scheduler": self.lr_scheduler,
            "power_t": self.power_t,
            "patience": self.patience,
            "factor": self.factor,
            "stoic_iter": self.stoic_iter,
            "epsilon": self.epsilon
        }

    def set_params(self, **params) -> "BasicRegressor":
        """
        Returns model's attribute that ready to set.

        ## Args:
            **params**: *dict*
            Model parameters to set.

        ## Returns:
            **BasicRegressor**: *The model instance with updated parameters.*

        ## Raises:
            **None**
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self