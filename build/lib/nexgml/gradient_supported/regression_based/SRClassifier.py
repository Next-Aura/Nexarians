# ========== LIBRARIES ==========
import numpy as np                              # Numpy for numerical computations
from scipy.sparse import issparse, spmatrix     # For sparse matrix handling
from typing import Literal, Optional            # More specific type hints
from nexgml.amo import forlinear                # For some math operation
from nexgml.indexing import (integer_labeling, 
                             one_hot_labeling)  # For indexing utilities
from warnings import warn                       # For warning messages
from nexgml.metrics import accuracy_score       # For accuracy metric
from nexgml.guardians import safe_array         # For numerical stability

# ========== THE MODEL ==========
class SRClassifier:
    """
    Stocastic Regression Classifier (SRC) is a linear classifier that uses gradient descent optimization with softmax for multi-class classification.
    It supports L1, L2, and Elastic Net regularization to prevent overfitting, and learning rate schedulers (constant, invscaling, plateau).
    Uses regression loss fucntion with gradient descent to minimize loss.
    Handle both dense and sparse input matrices.

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

      **classes**: *np.ndarray*
      Store unique classes from fitted data in fit() method.

      **n_classes**: *int*
      Number of unique class from data.

    ## Methods:
      **_calculate_loss(y_true, y_pred_proba)**: *Return float*
      Calculate model loss during training loop.

      **_calculate_grad(X, y)**: *Returns np.ndarray, float, np.ndarray*
      Calculate gradient of selected loss function, 
      return weight and bias gradient and also the linear combination.

      **predict_proba(X_test)**: *Return np.ndarray*
      Calculate class probability for classification.

      **fit(X_train, y_train)**: *Return None*
      Train model with inputed X_train and y_train argument data.

      **predict(X_test)**: *Return np.ndarray*
      Predict using weights from training session.

      **score(X_test, y_test)**: *Return float*
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
      >>> model = SRClassifier(loss='rmse')
      >>> model.fit(X_train, y_train)
      >>>
      >>> acc = model.score(X_test, y_test)
      >>> print("SRClassifier accuracy:", acc)
    ```
    """
    def __init__(
        self,  
        max_iter: int=1000, 
        learning_rate: float=0.01,
        penalty: Optional[Literal["l1", "l2", "elasticnet"]] | None="l2", 
        alpha: float=0.0001, 
        l1_ratio: float=0.5,
        loss: Literal["mse", "rmse", 'mae', 'smoothl1'] | None="mse",
        fit_intercept: bool=True, 
        tol: float=0.0001, 
        shuffle: bool | None=True, 
        random_state: int | None=None, 
        early_stopping: bool=True,
        verbose: int=0,
        verbosity: Literal['light', 'heavy'] | None = 'light',
        lr_scheduler: Literal["constant", "invscaling", "plateau", "adaptive"] | None="invscaling", 
        power_t: float=0.25, 
        patience: int=5, 
        factor: float=0.5, 
        delta: float=1.0,
        stoic_iter: int | None = 10,
        epsilon: float=1e-15
            ):
        """
        Initialize the Stocastic Regression Classifier model.

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

            **loss**: *{'mse', 'rmse', 'mae', 'smoothl1'}, default='mse'*
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

            **lr_scheduler**: *{'constant', 'invscaling', 'plateau', 'adaptive} or None, default='invscaling'*
            Strategy for learning rate adjustment over iterations.
            
            **power_t**: *float, default=0.25*
            The exponent for inverse scaling learning rate schedule (used if lr_scheduler='invscaling').

            **patience**: *int, default=5*
            Number of epochs to wait for loss improvement before reducing learning rate (used if lr_scheduler='plateau').

            **factor**: *float, default=0.5*
            Factor by which the learning rate will be reduced (used if lr_scheduler='plateau').

            **delta**: *float, default=1.0*
            Threshold for the Smooth L1 loss function.

            **stoic_iter**: *int or None, default=10*
            Number of initial epochs to skip before checking for convergence/tolerance in early stopping.
            
            **epsilon**: *float, default=1e-15*
            Small value to avoid numerical instability in calculations.

        ## Returns:
            **None**

        ## Raises:
            **ValueError**: *If invalid loss, penalty, or lr_scheduler type is provided.*
            **UserWarning**: *If verbose level 2 is used with heavy verbosity.*
        """
        # ========== PARAMETER VALIDATION ==========
        if penalty not in (None, "l1", "l2", "elasticnet"):
            raise ValueError(f"Invalid penalty argument {penalty}. Choose from 'l1', 'l2', or 'elasticnet'.")
        
        if verbosity not in ('light', 'heavy'):
            raise ValueError(f"Invalid verbosity argument, {verbosity}. Choose from 'light' or 'heavy'.")

        if loss not in ('mse', 'rmse', 'mae', 'smoothl1'):
            raise ValueError(f"Invalid loss argument, {loss}. Choose from 'mse', 'rmse', 'mae', or 'smoothl1'.")

        if lr_scheduler not in {'invscaling', 'constant', 'plateau', 'adaptive'}:
            raise ValueError(f"Invalid lr_scheduler argument {lr_scheduler}. Choose from 'invscaling', 'constant', 'plateau', or 'adaptive'.")
        
        if verbose == 2 and verbosity == 'heavy':
            warn("Verbose level 2 with heavy verbosity may produce excessive output.", UserWarning)

        # ========== HYPERPARAMETERS ==========
        self.max_iter = int(max_iter)              # Maximum number of training iterations (epochs)
        self.penalty = str(penalty)                # Regularization penalty type ('l1', 'l2', 'elasticnet', or None)
        self.lr_scheduler = str(lr_scheduler)      # Learning rate scheduler type ('invscaling', 'constant', 'plateau')
        self.learning_rate = float(learning_rate)  # Initial learning rate for gradient descent
        self.alpha = float(alpha)                  # Regularization strength (controls penalty magnitude)
        self.l1_ratio = float(l1_ratio)            # Elastic net mixing ratio between L1 and L2 (0 to 1)
        self.intercept = bool(fit_intercept)       # Whether to fit an intercept (bias) term
        self.tol = float(tol)                      # Tolerance for early stopping based on loss improvement
        self.power_t = float(power_t)              # Power parameter for inverse scaling learning rate scheduler
        self.shuffle = bool(shuffle)               # Whether to shuffle training data each epoch
        self.random_state = random_state           # Random seed for reproducible shuffling and initialization
        self.patience = int(patience)              # Number of epochs to wait before reducing learning rate (plateau)
        self.factor = float(factor)                # Factor by which to reduce learning rate on plateau
        self.early_stop = bool(early_stopping)     # Whether to enable early stopping
        self.verbose = int(verbose)                # Verbosity level for training progress logging (0: silent, 1: progress)
        self.stoic_iter = int(stoic_iter)          # Warm-up iterations before applying early stopping and lr scheduler
        self.loss = str(loss)                      # Loss function type
        self.delta = float(delta)                  # Huber loss threshold
        self.verbosity = str(verbosity)            # Verbosity level for logging
        self.epsilon = float(epsilon)              # Small constant to prevent division by zero in computations

        # ========== INTERNAL VARIABLES ==========
        self.weights = None                        # Model weights (coefficients) matrix of shape (n_features, n_classes)
        self.b = None                              # Bias term vector of shape (n_classes,)
        self.loss_history = []                     # List to store loss values for each training epoch
        self.classes = None                        # Array of unique class labels from training data
        self.n_classes = 0                         # Number of unique classes (determined during fit)
        self.current_lr = None                     # Current learning rate during training (updated by scheduler)
        self.best_loss = float('inf')              # Best loss achieved (used for plateau scheduler)
        self.wait = 0                              # Counter for epochs without improvement (plateau scheduler)


    # ========= HELPER METHODS =========
    def _calculate_loss(self, y_true: np.ndarray | spmatrix, y_pred_proba: np.ndarray | spmatrix) -> float:
        """
        Compute categorical cross-entropy loss with regularization.
        L1, L2, and Elastic Net regularization available.

        ## Args:
            **y_true**: *True one-hot encoded labels.*
            **y_pred_proba**: *Predicted class probabilities.*

        ## Returns:
            **float**: *Computed loss value.*

        ## Raises:
            **None**
        """
        y_pred_proba = np.clip(y_pred_proba, self.epsilon, 1 - self.epsilon)
        # Clip probabilities to avoid numerical issues
        
        # Calculate loss with MSE formula
        if self.loss == 'mse':
            loss = forlinear.mean_squared_error(y_true, y_pred_proba)
        
        # Calculate loss with RMSE formula
        elif self.loss == 'rmse':
            loss = forlinear.root_squared_error(y_true, y_pred_proba)
        
        # Calculate loss with MAE formula
        elif self.loss == 'mae':
            loss = forlinear.mean_absolute_error(y_true, y_pred_proba)

        elif self.loss == 'smoothl1':
            loss = forlinear.smoothl1_loss(y_true, y_pred_proba)

        # L1 regularization
        if self.penalty == 'l1':
            loss += forlinear.lasso(self.weights, self.alpha)
        
        # L2 regularization
        elif self.penalty == 'l2':
            loss += forlinear.ridge(self.weights, self.alpha)
        
        # Elastic Net regularization
        elif self.penalty == 'elasticnet':
            loss += forlinear.elasticnet(self.weights, self.alpha, self.l1_ratio)
        
        # Loss after regularization
        return loss

    def _calculate_grad(self, 
                        X: np.ndarray | spmatrix, 
                        y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate gradient of loss function with regulation.
        L1, L2, and Elastic Net available.
        
        ## Args:
            **X**: *np.ndarray* or *spmatrix*
            Input features.

            **y**: *np.ndarray*
            True target values.

        ## Return:
            **tuple**: *(np.ndarray, np.ndarray, np.ndarray).*
            np.ndarray: Gradient w.r.t. weights.
            np.ndarray: Gradient w.r.t. bias.
            np.ndarray: Calculated linear combination.
            
        ## Raises:
            **None**
        """

        # Ensure at least 2D for dense matrices
        if not issparse(X):
            X = np.atleast_2d(X)
        
        # Linear combination
        z = X @ self.weights
        if self.intercept:
            # Add bias if intercept is used
            z += self.b
        
        # Compute softmax probabilities
        y_pred_proba = forlinear.softmax(z)
        
        # Prediction error
        error = y_pred_proba - y

        # MSE loss gradient
        if self.loss == 'mse':
            grad_w, grad_b = forlinear.mse_deriv(X, error, self.intercept)
        
        # RMSE loss gradient
        elif self.loss == 'rmse':
           grad_w, grad_b = forlinear.rmse_deriv(X, error, self.intercept)
        
        # MAE loss gradient
        elif self.loss == 'mae':
           grad_w, grad_b = forlinear.mae_deriv(X, error, self.intercept)

        # Smooth L1 loss gradient
        elif self.loss == 'smoothl1':
           grad_w, grad_b = forlinear.smoothl1_deriv(X, error, self.intercept, self.delta)

        # L1 regularization gradient
        if self.penalty == 'l1':
            grad_w += forlinear.lasso_deriv(self.weights, self.alpha)

        # L2 regularization gradient
        elif self.penalty == 'l2':
            grad_w += forlinear.ridge_deriv(self.weights, self.alpha)
        
        # Elastic Net regularization
        elif self.penalty == 'elasticnet':
            grad_w += forlinear.elasticnet_deriv(self.weights, self.alpha, self.l1_ratio)
        
        # Return gradients for weights and bias
        return grad_w, grad_b, z
    
    def predict_proba(self, X_test: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict class probabilities using the trained model.

        ## Args:
            **X_test**: *np.ndarray* or *spmatrix*
            Input features for prediction.

        ## Returns: 
            **np.ndarray**: *Predicted class probabilities.*

        ## Raises:
            **ValueError**: *If model is not trained or weights are uninitialized.*
        """
        # Check if not sparse
        if not issparse(X_test):
            # Reshape 1D to 2D if X is 1D
            if X_test.ndim == 1:
                X_processed = X_test.reshape(-1, 1)

            else:
                # Or keep as is
                X_processed = X_test

            # Convert to float array
            X_processed = np.asarray(X_processed, dtype=np.float64)

        else:
            # Keep sparse
            X_processed = X_test
        
        # Check if model is trained
        if self.n_classes == 0:
             raise ValueError("Model not trained. Call fit() first.")
        
        # Check if weights initialized
        if self.weights is None:
            raise ValueError("Weights not initialized. Call fit() first.")
        
        # Linear combination
        z = X_processed @ self.weights
        
        # Add bias if enabled
        if self.intercept:
           z += self.b
        
        # Return softmax probabilities
        return forlinear.softmax(z)

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
            **ValueError**: *If input data contains NaN/Inf or dimensions mismatch.*
            **OverflowError**: *If parameters (weight, bias or loss) become infinity during training loop.*
            **RuntimeWarning**: *If overflow is detected and values are clipped.*
        """
        # ---------- Preprocess input data ----------
        # Check if not sparse
        if not issparse(X_train):
            if X_train.ndim == 1:
                # Reshape 1D to 2D if X is 1D
                X_processed = X_train.reshape(-1, 1)

            else:
                # Or keep as is
                X_processed = X_train
            
            # Convert to numpy array
            X_processed = np.asarray(X_processed)

        else:
            # Keep sparse (CSR or CSC depend on the shape)
            if X_train.shape[0] > X_train.shape[1]:
              X_processed = X_train.tocsr()

            else:
              X_processed = X_train.tocsc()
        
        # Get dimensions
        num_samples, num_features = X_processed.shape
        
        # Ensure y is 1D array
        y_processed = np.asarray(y_train).ravel()

        # Check sparse data for NaN/Inf (inspect .data) or dense arrays appropriately
        if issparse(X_processed):
            if not np.all(np.isfinite(X_processed.data)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values in its data. Please clean your data.")
        else:
            if not np.all(np.isfinite(X_processed)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")
        
        # Check y for NaN/Inf
        if not np.all(np.isfinite(y_processed)):
            raise ValueError("Input target (y_train) contains NaN or Infinity values. Please clean your data.")
        
        # Check sample count match
        if X_processed.shape[0] != y_processed.shape[0]:
            raise ValueError(
                f"Number of samples in X_train ({X_processed.shape[0]}) "
                f"must match number of samples in y_train ({y_processed.shape[0]})."
            )
        
        # Unique classes (preserve original label values)
        self.classes = np.unique(y_processed)
        # Number of classes
        self.n_classes = len(self.classes)

        # Check for at least 2 classes
        if self.n_classes < 2:
            raise ValueError("Class label must have at least 2 types.")

        # Map arbitrary class labels to integer indices [0..n_classes-1]
        # Use integer_labeling helper to produce integer labels for downstream operations
        y_int = integer_labeling(y_processed, self.classes, to_integer_from='labels')

        # Get one-hot from integer labels
        y_onehot = one_hot_labeling(y_int, np.arange(self.n_classes))

        # ---------- Pre-train process ----------
        # Initialize weights if needed
        if self.weights is None or self.weights.shape != (num_features, self.n_classes):
            # Random normal init
            rng = np.random.default_rng(self.random_state)
            self.weights = rng.normal(0, 0.01, (num_features, self.n_classes))

        self.b = np.zeros(self.n_classes)                   # Initialize bias

        # RNG for shuffling
        rng = np.random.default_rng(self.random_state)
        self.current_lr = self.learning_rate
        # Iteration loop
        for i in range(self.max_iter):
            # Shuffle data if enabled
            if self.shuffle:
                indices = rng.permutation(num_samples)  # Permutation indices
                X_processed = X_processed[indices]      # Shuffle X
                y_onehot = y_onehot[indices]            # Shuffle y
            
            if i > self.stoic_iter:
                if self.lr_scheduler == 'constant':
                    # Keep learning rate constant
                    self.current_lr = self.learning_rate

                elif self.lr_scheduler == 'invscaling':
                    # Inverse scaling decay
                    self.current_lr = self.learning_rate / ((i + 1)**self.power_t + self.epsilon)
                
                elif self.lr_scheduler == 'adaptive':
                    # Adaptive learning rate based on loss ratio
                    self.current_lr = min(10.0, max(self.current_lr * (self.loss_history[-1] / self.loss_history[-2]), 1e-8)) if i > 1 else self.current_lr

                elif self.lr_scheduler == 'plateau':
                    # Compute full dataset loss
                    current_loss = self._calculate_loss(y_onehot, self.predict_proba(X_processed))
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

            # Compute gradients using current proba
            grad_w, grad_b, z_current = self._calculate_grad(X_processed, y_onehot)

            # Get logits probabilities with softmax
            y_proba_current = forlinear.softmax(z_current)

            # Current loss
            loss = self._calculate_loss(y_onehot, y_proba_current)

            if np.isnan(loss):
                loss = safe_array(loss)

            # Store loss
            self.loss_history.append(loss)

            # Update weights
            self.weights -= self.current_lr * grad_w

            if self.intercept:
                # Update bias if intercept if used
                self.b -= self.current_lr * grad_b

            # Check for NaN/Inf during training loop
            if not np.all(np.isfinite(self.weights)) or (self.intercept and not np.all(np.isfinite(self.b))):
                self.weights = safe_array(self.weights)
                
                if self.intercept:
                    self.b = safe_array(self.b)

            # Check for NaN/Inf during training loop
            if not np.all(np.isfinite(self.weights)) or (self.intercept and not np.all(np.isfinite(self.b))):
                raise OverflowError(f"Weights or bias became NaN/Inf at epoch {i + 1}. Stopping training early.")

            # Check loss for NaN/Inf during training loop
            if not np.isfinite(loss):
                raise OverflowError(f"Loss became NaN/Inf at epoch {i + 1}. Stopping training early.")

            # Light verbose logging
            if self.verbose == 1 and ((i % max(1, self.max_iter // 20)) == 0 or i < 5) and self.verbosity == 'light':
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {np.mean(self.b):.6f}")

            elif self.verbose == 2 and self.verbosity == 'light':
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {np.mean(self.b):.6f}")

            # Heavy verbose logging
            if self.verbose == 1 and ((i % max(1, self.max_iter // 20)) == 0 or i < 5) and self.verbosity == 'heavy':
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {loss:.8f}, Avg Weights: {np.mean(self.weights):.8f}, Avg Bias: {np.mean(self.b):.8f}, Current LR: {self.current_lr:.8f}")

            elif self.verbose == 2 and self.verbosity == 'heavy':
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {loss:.8f}, Avg Weights: {np.mean(self.weights):.8f}, Avg Bias: {np.mean(self.b):.8f}, Current LR: {self.current_lr:.8f}")

            # ========== EARLY STOPPING ==========
            if self.early_stop and i > self.stoic_iter:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                    break 
                
                if i > 2 * self.stoic_iter:
                    if abs(np.mean(self.loss_history[-self.stoic_iter:]) - np.mean(self.loss_history[-2*self.stoic_iter:-self.stoic_iter])) < self.tol:
                        break

    def predict(self, X_test: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict class labels using the trained model.

        ## Args:
            **X_test**: *np.ndarray* or *spmatrix*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *Predicted class labels.*

        ## Raises:
            **ValueError**: *If model is not trained or weights are uninitialized.*
        """
        # Get probabilities
        probas = self.predict_proba(X_test)

        # Get class indices
        pred_class = np.argmax(probas, axis=1)
        
        # Map to original class labels
        if self.classes is not None and len(self.classes) == self.n_classes:
            pred_class = np.array([self.classes[idx] for idx in pred_class])
        
        # Return predictions
        return pred_class
    
    def score(self, X_test: np.ndarray | spmatrix, y_test: np.ndarray) -> float:
        """
        Calculate the mean accuracy on the given test data and labels.

        ## Args:
            **X_test**: *np.ndarray* or *spmatrix*
            Feature matrix.

            **y_test**: *np.ndarray*
            True target labels.

        ## Returns:
            **float**: *Mean accuracy score.*

        ## Raises:
            **None**
        """
        # ========== PREDICTION ==========
        y_pred = self.predict(X_test)
        
        # ========== ACCURACY CALCULATION ==========
        return accuracy_score(y_test, y_pred)
    
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
            "delta": self.delta,
            "stoic_iter": self.stoic_iter,
            "epsilon": self.epsilon
        }

    def set_params(self, **params) -> "SRClassifier":
        """
        Returns model's attribute that ready to set.

        ## Args:
            **params**: *dict*
            Model parameters to set.

        ## Returns:
            **SRClassifier**: *The model instance with updated parameters.*

        ## Raises:
            **None**
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self