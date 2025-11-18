# ========== LIBRARIES ==========
import numpy as np                             # For numerical operations
from scipy.sparse import issparse, spmatrix    # For sparse matrix support
from typing import Literal, Optional           # More specific type hints
from nexgml.amo import forlinear               # For specific numerical operations
from nexgml.indexing import one_hot_labeling   # For one-hot labeling

# ========== THE MODEL ==========
class IntenseClassifier:
    """
    Gradient Supported Intense Classifier (GSIC) is an advanced linear classifier that uses mini-batch gradient descent optimization with softmax for multi-class classification.
    It supports L1, L2, and Elastic Net regularization to prevent overfitting, along with multiple optimizers (MBGD, Adam, AdamW) and learning rate schedulers (constant, invscaling, plateau).
    Categorical cross-entropy loss is used.
    Supports sparse matrices for memory efficiency and includes early stopping for robust training.
    """
    def __init__(
        self,  
        max_iter: int=1000, 
        learning_rate: float=0.01,
        penalty: Optional[Literal["l1", "l2", "elasticnet"]] | None="l2", 
        alpha: float=0.0001, 
        l1_ratio: float=0.5, 
        fit_intercept: bool=True, 
        tol: float=0.0001, 
        shuffle: bool | None=True, 
        random_state: int | None=None, 
        early_stopping: bool=True,
        verbose: int=0,
        lr_scheduler: Literal["constant", "invscaling", 'plateau'] | None='invscaling', 
        optimizer: Literal['mbgd', 'adam', 'adamw'] | None='mbgd', 
        batch_size: int=16, 
        power_t: float=0.25, 
        patience: int=5, 
        factor: float=0.5, 
        stoic_iter: int | None = 10
            ):
        """
        Initialize the SoftIntenseClassifier model.

        ## Args:
            **max_iter**: *int, default=1000*
            Maximum number of training iterations (epochs).

            **learning_rate**: *float, default=0.01*
            Initial step size for optimizer updates.

            **penalty**: *{'l1', 'l2', 'elasticnet'} or None, default='l2'*
            Type of regularization ('l1', 'l2', 'elasticnet') or None.

            **alpha**: *float, default=0.0001*
            Regularization strength (used if penalty is not None).

            **l1_ratio**: *float, default=0.5*
            Mixing parameter for elastic net (0 <= l1_ratio <= 1).

            **fit_intercept**: *bool, default=True*
            If True, include a bias term (intercept).

            **tol**: *float, default=0.0001* 
            Tolerance for early stopping based on loss convergence.

            **shuffle**: *bool, default=True*
            If True, shuffle data each epoch.

            **random_state**: *float, default=None*
            Seed for random number generator for reproducibility.

            **early_stopping**: *bool, default=True*
            If true, will make the model end the training loop early if the model performance plateaus.

            **verbose**: *int, default=0*
            If 1, print training progress (epoch, loss, etc.).
            
            **lr_scheduler**: *{'constant', 'invscaling', 'plateau'} or None, default='invscaling'*
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

            **stoic_iter**: *int or None, default=10*
            Number of initial epochs to skip before checking for convergence/tolerance in early stopping.

        ## Returns:
            **None**

        ## Raises:
            **ValueError**: *If invalid penalty, optimizer, or lr_scheduler type is provided, 
            or if AdamW is used with a non-L2 penalty.*
        """
        # ========== PARAMETER VALIDATION ==========
        if penalty not in {'l1', 'l2', 'elasticnet', None}:
            raise ValueError(f"Invalid penalty argument {penalty}. Choose from 'l1', 'l2', 'elasticnet', or None.")
        
        if lr_scheduler not in {'invscaling', 'constant', 'plateau'}:
            raise ValueError(f"Invalid lr_scheduler argument {lr_scheduler}. Choose from 'invscaling', 'constant', or 'plateau'.")
        
        if optimizer not in {'mbgd', 'adam', 'adamw'}:
            raise ValueError(f"Invalid optimizer argument {optimizer}. Choose from 'mbgd', 'adam', or 'adamw'.")
        
        if optimizer == 'adamw' and penalty not in {'l2', None}:
            raise ValueError("AdamW optimizer only supports L2 regularization or no regularization.")

        # ========== HYPERPARAMETERS ==========
        self.max_iter = int(max_iter)              # Maximum number of training iterations (epochs)
        self.penalty = penalty                     # Regularization penalty type ('l1', 'l2', 'elasticnet', or None)
        self.lr_scheduler = lr_scheduler           # Learning rate scheduler type ('invscaling', 'constant', 'plateau')
        self.learning_rate = float(learning_rate)  # Initial learning rate for gradient descent
        self.alpha = float(alpha)                  # Regularization strength (controls penalty magnitude)
        self.l1_ratio = float(l1_ratio)            # Elastic net mixing ratio between L1 and L2 (0 to 1)
        self.intercept = bool(fit_intercept)       # Whether to fit an intercept (bias) term
        self.tol = float(tol)                      # Tolerance for early stopping based on loss improvement
        self.power_t = float(power_t)              # Power parameter for inverse scaling learning rate scheduler
        self.batch_size = int(batch_size)          # Number of samples per batch for mini-batch gradient descent
        self.shuffle = bool(shuffle)               # Whether to shuffle training data each epoch
        self.random_state = random_state           # Random seed for reproducible shuffling and initialization
        self.optimizer = optimizer                 # Optimization algorithm ('mbgd', 'adam', 'adamw')
        self.patience = int(patience)              # Number of epochs to wait before reducing learning rate (plateau)
        self.factor = float(factor)                # Factor by which to reduce learning rate on plateau
        self.early_stop = bool(early_stopping)     # Whether to enable early stopping
        self.verbose = int(verbose)                # Verbosity level for training progress logging (0: silent, 1: progress)
        self.stoic_iter = int(stoic_iter)          # Warm-up iterations before applying early stopping and lr scheduler

        # ========== INTERNAL VARIABLES ==========
        self.epsilon = 1e-15                       # Small constant to prevent division by zero in computations
        self.weights = None                        # Model weights (coefficients) matrix of shape (n_features, n_classes)
        self.b = None                              # Bias term vector of shape (n_classes,)
        self.loss_history = []                     # List to store loss values for each training epoch
        self.classes = None                        # Array of unique class labels from training data
        self.n_classes = 0                         # Number of unique classes (determined during fit)
        self.current_lr = None                     # Current learning rate during training (updated by scheduler)
        self.best_loss = float('inf')              # Best loss achieved (used for plateau scheduler)
        self.wait = 0                              # Counter for epochs without improvement (plateau scheduler)

        # ---------- Adam/AdamW specific ----------
        if optimizer in ('adam', 'adamw'):
            self.m_w = None                            # First moment estimate for weights (Adam/AdamW optimizer)
            self.v_w = None                            # Second moment estimate for weights (Adam/AdamW optimizer)
            self.m_b = None                            # First moment estimate for bias (Adam/AdamW optimizer)
            self.v_b = None                            # Second moment estimate for bias (Adam/AdamW optimizer)
            self.beta1 = 0.9                           # Exponential decay rate for first moment estimates
            self.beta2 = 0.999                         # Exponential decay rate for second moment estimates

    # ========= HELPER METHODS =========
    def _calculate_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float: 
        """
        Compute categorical cross-entropy loss with regularization.

        Note: This implementation uses mean cross-entropy.
        L1, L2, and Elastic Net regularization are supported.
        AdamW (L2) is handled separately in the update step.

        ## Args:
            **y_true**: *np.ndarray*
            True one-hot encoded labels.
            
            **y_pred_proba**: *np.ndarray*
            Predicted class probabilities.

        ## Returns:
            **float**: *Computed loss value.*

        ## Raises:
            **None**
        """

        loss = forlinear.categorical_ce(y_true, y_pred_proba, mean=True)

        # L1 regularization
        if self.penalty == "l1":
            loss += forlinear.lasso(self.weights, self.alpha)

        # L2 regularization
        elif self.penalty == "l2" and self.optimizer != 'adamw':
            loss += forlinear.ridge(self.weights, self.alpha)
        
        # Elastic Net
        elif self.penalty == 'elasticnet':
            loss += forlinear.elasticnet(self.weights, self.alpha, self.l1_ratio)

        return loss

    def _calculate_grad(self, X: np.ndarray | spmatrix, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients of the categorical cross-entropy loss with respect to weights and bias.

        Supports both dense and sparse matrices for efficient computation.

        ## Args:
            **X**: *np.ndarray or spmatrix*
            Input features for the batch.

            **y**: *np.ndarray*
            True one-hot encoded labels for the batch.
            
        ## Returns:
            **tuple**: *(np.ndarray, np.ndarray, np.ndarray).*
            np.ndarray: Gradient w.r.t. weights.
            np.ndarray: Gradient w.r.t. bias.
            np.ndarray: Calculated linear combination.
            
        ## Raises:
            **None**
        """
        if not issparse(X):
            X = np.atleast_2d(X)
            # Ensure at least 2D for dense matrices
        z = X @ self.weights
        # Linear combination of inputs and weights
        if self.intercept:
            z += self.b
            # Add bias term
        y_pred_proba = forlinear.softmax(z)
        # Compute softmax probabilities
        error = y_pred_proba - y
        # Prediction error (residuals)
        grad_w, grad_b = forlinear.cce_deriv(X, error, self.intercept, self.n_classes)
        # Bias gradient (mean error per class)

        if self.penalty == 'l1':
            grad_w += forlinear.lasso_deriv(self.weights, self.alpha)

        elif self.penalty == 'l2':
            grad_w += forlinear.ridge_deriv(self.weights, self.alpha)

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
            **ValueError**: *If model is not trained.*
        """
        if not issparse(X_test):
            if X_test.ndim == 1:
                X_processed = X_test.reshape(-1, 1)
                # Reshape 1D to 2D
            X_processed = np.asarray(X_test, dtype=np.float64)
            # Convert to float64 numpy array
        else:
            X_processed = X_test
            # Keep sparse matrix as is

        if self.n_classes == 0:
            raise ValueError("Model not trained. Call fit() first.")

        z = X_processed @ self.weights
        # Linear combination
        if self.intercept:
            z += self.b
            # Add bias if intercept is used
        return forlinear.softmax(z)
        # Return softmax probabilities

    # ========= MAIN METHODS =========
    def fit(self, X_train, y_train):
        """
        Fit the model to the training data using mini-batch gradient descent.
        Supports multiple optimizers (MBGD, Adam, AdamW), learning rate schedules,
        and early stopping.
        
        ## Args:
            **X_train**: *np.ndarray or spmatrix*
            Training input features.

            **y_train**: *np.ndarray or spmatrix*
            Training target values.
            
        ## Returns:
            **None**
            
        ## Raises:
            **ValueError**: *If input data contains NaN/Inf, dimensions mismatch, or < 2 classes.*
        """
        # ========== DATA PREPROCESSING ==========
        if not issparse(X_train):
            # Check if input is sparse matrix
            if X_train.ndim == 1:
                # If 1D array, reshape to 2D
                X_train = X_train.reshape(-1, 1)
                # Reshape for single feature
            X_processed = np.asarray(X_train, dtype=np.float64)
            # Convert to float64 numpy array
        
        # Keep sparse matrix as is (CSR or CSC)
        else:
            if X_train.shape[0] > X_train.shape[1]:
              X_processed = X_train.tocsr()

            else:
              X_processed = X_train.tocsc()

        y_processed = np.asarray(y_train, dtype=np.float64).ravel()
        # Convert y to 1D float64 array

        # ========== DATA VALIDATION ==========
        if issparse(X_processed):
            # Check sparse data for NaN/Inf
            if not np.all(np.isfinite(X_processed.data)):
                # Ensure all sparse data is finite
                raise ValueError("Input features (X_train) contain NaN or Infinity values.")
        else:
            # Check dense data for NaN/Inf
            if not np.all(np.isfinite(X_processed)):
                # Ensure all dense data is finite
                raise ValueError("Input features (X_train) contain NaN or Infinity values.")

        if not np.all(np.isfinite(y_processed)):
            # Check y for NaN/Inf
            raise ValueError("Input target (y_train) contains NaN or Infinity values.")

        if X_processed.shape[0] != y_processed.shape[0]:
            # Check sample count match
            raise ValueError(f"X_train ({X_processed.shape[0]}) and y_train ({y_processed.shape[0]}) sample mismatch.")

        num_samples, num_features = X_processed.shape
        # Get data dimensions
        self.classes = np.unique(y_processed)
        # Extract unique class labels
        self.n_classes = len(self.classes)
        # Number of classes
        self.loss_history = []
        # Initialize loss history list

        if self.n_classes < 2:
            # Ensure at least 2 classes
            raise ValueError("Class label must have at least 2 types.")
        
        if self.weights is None or self.weights.shape != (num_features, self.n_classes):
            # Initialize weights if needed
            self.weights = np.zeros((num_features, self.n_classes), dtype=np.float64)
            # Zero initialization for weights

        if self.intercept and (self.b is None or self.b.shape != (self.n_classes,)):
            # Initialize bias if needed
            self.b = np.zeros(self.n_classes, dtype=np.float64)
            # Zero initialization for bias

        y_one_hot = one_hot_labeling(y_processed, self.classes)
        # Data label one-hot transform

        if self.optimizer in {'adam', 'adamw'}:
            # Initialize Adam/AdamW moments if needed
            self.m_w = np.zeros_like(self.weights)
            # First moment for weights
            self.v_w = np.zeros_like(self.weights)
            # Second moment for weights
            self.m_b = np.zeros_like(self.b) if self.intercept else None
            # First moment for bias
            self.v_b = np.zeros_like(self.b) if self.intercept else None
            # Second moment for bias
        
        rng = np.random.default_rng(self.random_state)
        # Random number generator for shuffling
        num_batches = int(np.ceil(num_samples / self.batch_size))
        # Calculate number of batches
        self.current_lr = self.learning_rate
        # Initialize current learning rate
        self.best_loss = float('inf')
        # Initialize best loss for early stopping
        self.wait = 0
        # Initialize wait counter for early stopping
        
        # ========== TRAINING LOOP ==========
        for i in range(self.max_iter):
            # ========== LEARNING RATE SCHEDULING ==========
            # Apply LR scheduler after warm-up iterations
            if i > self.stoic_iter:
                if self.lr_scheduler == 'constant':
                    self.current_lr = self.learning_rate
                    # Keep learning rate constant

                elif self.lr_scheduler == 'invscaling':
                    self.current_lr = self.learning_rate / ((i + 1)**self.power_t + self.epsilon)
                    # Inverse scaling decay

                elif self.lr_scheduler == 'plateau':
                    current_loss = self._calculate_loss(y_one_hot, self.predict_proba(X_processed))
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

            # ========== DATA SHUFFLING ==========
            if self.shuffle:
                indices = rng.permutation(num_samples)
                # Generate random permutation indices
                X_shuffled = X_processed[indices] if not issparse(X_processed) else X_processed[indices]
                # Shuffle X
                y_shuffled = y_one_hot[indices]
                # Shuffle y
            else:
                X_shuffled = X_processed
                # No shuffling
                y_shuffled = y_one_hot

            # ========== BATCH PROCESSING ==========
            epoch_loss_sum = 0.0
            for j in range(num_batches):
                s_idx = j * self.batch_size
                # Start index for current batch
                e_idx = min((j + 1) * self.batch_size, num_samples)
                # End index for current batch
                X_batch = X_shuffled[s_idx:e_idx]
                # Extract batch features
                y_batch = y_shuffled[s_idx:e_idx]
                # Extract batch labels
                grad_w, grad_b, z_batch = self._calculate_grad(X_batch, y_batch)
                # Compute gradients for batch

                # ========== PARAMETER UPDATES ==========
                if self.optimizer == 'mbgd':
                    self.weights -= self.current_lr * grad_w
                    # Update weights using MBGD
                    if self.intercept:
                        self.b -= self.current_lr * grad_b
                        # Update bias using MBGD

                elif self.optimizer == 'adam':
                    t = i * num_batches + j + 1
                    # Time step for Adam
                    self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_w
                    # Update first moment for weights
                    self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_w**2)
                    # Update second moment for weights
                    m_w_hat = self.m_w / (1 - self.beta1**t)
                    # Bias-corrected first moment
                    v_w_hat = self.v_w / (1 - self.beta2**t)
                    # Bias-corrected second moment
                    self.weights -= self.current_lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                    # Update weights

                    if self.intercept:
                        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
                        # Update first moment for bias
                        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b**2)
                        # Update second moment for bias
                        m_b_hat = self.m_b / (1 - self.beta1**t)
                        # Bias-corrected first moment
                        v_b_hat = self.v_b / (1 - self.beta2**t)
                        # Bias-corrected second moment
                        self.b -= self.current_lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
                        # Update bias
                elif self.optimizer == 'adamw':
                    t = i * num_batches + j + 1
                    # Time step for AdamW
                    self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_w
                    # Update first moment for weights
                    self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_w**2)
                    # Update second moment for weights
                    m_w_hat = self.m_w / (1 - self.beta1**t)
                    # Bias-corrected first moment
                    v_w_hat = self.v_w / (1 - self.beta2**t)
                    # Bias-corrected second moment
                    self.weights -= self.current_lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                    # Update weights

                    if self.penalty == "l2":
                        self.weights -= self.current_lr * self.alpha * self.weights
                        # Apply L2 weight decay

                    if self.intercept:
                        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
                        # Update first moment for bias
                        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b**2)
                        # Update second moment for bias
                        m_b_hat = self.m_b / (1 - self.beta1**t)
                        # Bias-corrected first moment
                        v_b_hat = self.v_b / (1 - self.beta2**t)
                        # Bias-corrected second moment
                        self.b -= self.current_lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
                        # Update bias
                else:
                    raise ValueError(f"Optimizer '{self.optimizer}' not supported.")

                # ========== BATCH LOSS COMPUTATION ==========
                y_proba_batch = forlinear.softmax(z_batch)
                # Compute softmax probabilities
                epoch_loss_sum += self._calculate_loss(y_batch, y_proba_batch)
                # Accumulate batch loss

            # ========== EPOCH LOSS AND LOGGING ==========
            # Average loss over all batches
            avg_epoch_loss = epoch_loss_sum / num_batches
            # Store epoch loss
            self.loss_history.append(avg_epoch_loss)
            # Level 1 verbose logging
            if self.verbose == 1 and ((i % max(1, self.max_iter // 20)) == 0 or i < 5):
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {avg_epoch_loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {np.mean(self.b):.6f}, Learning Rate: {self.current_lr:.6f}")

            # Level 2 verbose logging
            elif self.verbose == 2:
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {avg_epoch_loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {np.mean(self.b):.6f}, Learning Rate: {self.current_lr:.6f}")

            # ========== EARLY STOPPING ==========
            if self.early_stop and i > self.stoic_iter:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
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
            **ValueError**: *If model is not trained (propagated from predict_proba).*
        """
        probas = self.predict_proba(X_test)
        # Get predicted probabilities
        pred_class = np.argmax(probas, axis=1)
        # Choose class with highest probability

        if self.classes is not None and len(self.classes) == self.n_classes:
            pred_class = np.array([self.classes[idx] for idx in pred_class])
            # Map indices to original classes
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
        # Compare prediction with true labels and compute mean
        return np.mean(y_pred == y_test)
    
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
            "stoic_iter": self.stoic_iter
        }

    def set_params(self, **params) -> 'IntenseClassifier':
        """
        Returns model's attribute that ready to set.

        ## Args:
            **params**: *dict*
            Model parameters to set.

        ## Returns:
            **IntenseClassifier**: *The model instance with updated parameters.*

        ## Raises:
            **None**
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self