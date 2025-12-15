# ========== LIBRARIES ==========
import numpy as np                             # For numerical operations
from scipy.sparse import issparse, spmatrix    # For sparse matrix support
from typing import Literal, Optional           # More specific type hints
from nexgml.amo import forlinear               # For specific numerical operations
from nexgml.indexing import one_hot_labeling   # For one-hot labeling
from warnings import warn                      # For warning messages
from nexgml.metrics import accuracy_score      # For accuracy metric
from nexgml.guardians import safe_array        # For numerical stability

# ========== THE MODEL ==========
class IntenseClassifier:
    """
    Gradient Supported Intense Classifier (GSIC) is an advanced linear classifier that uses mini-batch gradient descent optimization with softmax for multi-class classification.
    It supports L1, L2, and Elastic Net regularization to prevent overfitting, along with multiple optimizers (MBGD, Adam, AdamW) and learning rate schedulers (constant, invscaling, plateau, adaptive).
    Categorical cross-entropy loss is used.
    Supports sparse matrices for memory efficiency and includes early stopping for robust training.
    
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

      ### If Adam or AdamW optimizer is used:
        **m_w**: *np.ndarray*
        Store first moment estimate for weights.

        **v_w**: *np.ndarray*
        Store second moment estimate for weights.

        **m_b**: *np.ndarray*
        Store first moment estimate for bias.

        **v_b**: *np.ndarray*
        Store second moment estimate for bias.

        **beta1**: *float*
        Exponential decay rate for first moment.

        **beta2**: *float*
        Exponential decay rate for second moment.

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
      >>> model = IntenseClassifier(optimizer='adamw')
      >>> model.fit(X_train, y_train)
      >>>
      >>> acc = model.score(X_test, y_test)
      >>> print("IntenseClassifier accuracy:", acc)
    ```
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
        verbosity: Literal['light', 'heavy'] | None = 'light',
        lr_scheduler: Literal["constant", "invscaling", "plateau", "adaptive"] | None="invscaling", 
        optimizer: Literal["mbgd", "adam", "adamw"] | None="mbgd", 
        batch_size: int=16, 
        power_t: float=0.25, 
        patience: int=5, 
        factor: float=0.5, 
        stoic_iter: int | None = 10,
        epsilon: float=1e-15,
        adalr_window: int=5,
        w_init_scale: float=0.01
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

            **stoic_iter**: *int or None, default=10*
            Number of initial epochs to skip before checking for convergence/tolerance in early stopping.
            
            **epsilon**: *float, default=1e-15*
            Small value to avoid numerical instability in calculations.

            **adalr_window**: *int, default=5*
            Loss window for 'adaptive' learning rate (AdaLR) scheduler.

            **w_init_scale**: *float, default=0.01*
            Weight initialization scale.

        ## Returns:
            **None**

        ## Raises:
            **ValueError**: *If invalid penalty, verbosity, optimizer, or lr_scheduler type is provided, 
            or if AdamW is used with a non-L2 penalty.*
            **UserWarning**: *If verbose level 2 is used with heavy verbosity.*
        """
        # ========== PARAMETER VALIDATION ==========
        if penalty not in {'l1', 'l2', 'elasticnet', None}:
            raise ValueError(f"Invalid penalty argument {penalty}. Choose from 'l1', 'l2', 'elasticnet', or None.")
        
        if lr_scheduler not in {'invscaling', 'constant', 'plateau', 'adaptive'}:
            raise ValueError(f"Invalid lr_scheduler argument {lr_scheduler}. Choose from 'invscaling', 'constant', 'plateau', or 'adaptive'.")
        
        if verbosity not in ('light', 'heavy'):
            raise ValueError(f"Invalid verbosity argument, {verbosity}. Choose from 'light' or 'heavy'.")

        if optimizer not in {'mbgd', 'adam', 'adamw'}:
            raise ValueError(f"Invalid optimizer argument {optimizer}. Choose from 'mbgd', 'adam', or 'adamw'.")
        
        if optimizer == 'adamw' and penalty not in {'l2', None}:
            raise ValueError("AdamW optimizer only supports L2 regularization or no regularization.")

        if verbose == 2 and verbosity == 'heavy':
            warn("Verbose level 2 with heavy verbosity may produce excessive output.", UserWarning)

        # ========== HYPERPARAMETERS ==========
        self.max_iter = int(max_iter)              # Maximum number of training iterations (epochs)
        self.penalty = str(penalty)                # Regularization penalty type ('l1', 'l2', 'elasticnet', or None)
        self.lr_scheduler = str(lr_scheduler)      # Learning rate scheduler type ('invscaling', 'constant', 'plateau')
        self.learning_rate = np.float32(learning_rate)  # Initial learning rate for gradient descent
        self.alpha = float(alpha)                  # Regularization strength (controls penalty magnitude)
        self.l1_ratio = np.float32(l1_ratio)       # Elastic net mixing ratio between L1 and L2 (0 to 1)
        self.intercept = bool(fit_intercept)       # Whether to fit an intercept (bias) term
        self.tol = np.float32(tol)                 # Tolerance for early stopping based on loss improvement
        self.power_t = np.float32(power_t)         # Power parameter for inverse scaling learning rate scheduler
        self.batch_size = np.int32(batch_size)     # Number of samples per batch for mini-batch gradient descent
        self.shuffle = bool(shuffle)               # Whether to shuffle training data each epoch
        self.random_state = random_state           # Random seed for reproducible shuffling and initialization
        self.optimizer = str(optimizer)            # Optimization algorithm ('mbgd', 'adam', 'adamw')
        self.patience = int(patience)              # Number of epochs to wait before reducing learning rate (plateau)
        self.factor = np.float32(factor)           # Factor by which to reduce learning rate on plateau
        self.early_stop = bool(early_stopping)     # Whether to enable early stopping
        self.verbose = int(verbose)                # Verbosity level for training progress logging (0: silent, 1: progress)
        self.stoic_iter = int(stoic_iter)          # Warm-up iterations before applying early stopping and lr scheduler
        self.verbosity = str(verbosity)            # Verbosity level for logging
        self.epsilon = np.float32(epsilon)         # Small constant to prevent division by zero in computations
        self.window = int(adalr_window)            # AdaLR loss window
        self.w_input = np.float32(w_init_scale)    # Weight initialize scale

        # ========== INTERNAL VARIABLES ==========
        self.weights = None                        # Model weights (coefficients) matrix of shape (n_features, n_classes)
        self.b = None                              # Bias term vector of shape (n_classes,)
        self.loss_history = []                     # List to store loss values for each training epoch
        self.classes = None                        # Array of unique class labels from training data
        self.n_classes = 0                         # Number of unique classes (determined during fit)
        self.current_lr = None                     # Current learning rate during training (updated by scheduler)
        self.best_loss = np.float32(np.inf)        # Best loss achieved (used for plateau scheduler)
        self.wait = 0                              # Counter for epochs without improvement (plateau scheduler)

        # ---------- Adam/AdamW specific ----------
        if optimizer in ('adam', 'adamw'):
            self.m_w = None                            # First moment estimate for weights (Adam/AdamW optimizer)
            self.v_w = None                            # Second moment estimate for weights (Adam/AdamW optimizer)
            self.m_b = None                            # First moment estimate for bias (Adam/AdamW optimizer)
            self.v_b = None                            # Second moment estimate for bias (Adam/AdamW optimizer)
            self.beta1 = np.float32(0.9)               # Exponential decay rate for first moment estimates
            self.beta2 = np.float32(0.999)             # Exponential decay rate for second moment estimates

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
        # Weight and bias gradient

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
        X_test = np.asarray(X_test)
        if not issparse(X_test):
            if X_test.ndim == 1:
                X_processed = X_test.reshape(-1, 1)
                # Reshape 1D to 2D

            else:
                X_processed = X_test
        else:
            X_processed = X_test

        X_processed = X_processed.astype(np.float32)

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
    def fit(self, X_train: np.ndarray | spmatrix, y_train: np.ndarray) -> None:
        """
        Fit the model to the training data using mini-batch gradient descent.
        Supports multiple optimizers (MBGD, Adam, AdamW), learning rate schedules,
        and early stopping.
        
        ## Args:
            **X_train**: *np.ndarray or spmatrix*
            Training input features.

            **y_train**: *np.ndarray*
            Training target values.
            
        ## Returns:
            **None**
            
        ## Raises:
            **ValueError**: *If input data contains NaN/Inf, dimensions mismatch, or < 2 classes.*
            **OverflowError**: *If numerical overflow occurs during training.*
            **RuntimeWarning**: *If overflow is detected and values are clipped.*
        """
        # ========== DATA PREPROCESSING ==========
        if not issparse(X_train):
            # Check if input is sparse matrix
            if X_train.ndim == 1:
                # If 1D array, reshape to 2D
                X_train = X_train.reshape(-1, 1)
                # Reshape for single feature
            X_processed = np.asarray(X_train, dtype=np.float32)
            # Convert to float32 numpy array
        
        # Keep sparse matrix as is (CSR or CSC)
        else:
            if X_train.shape[0] > X_train.shape[1]:
              X_processed = X_train.tocsr().astype(np.float32)

            else:
              X_processed = X_train.tocsc().astype(np.float32)

        y_processed = np.asarray(y_train, dtype=np.float32).ravel()
        # Convert y to 1D float32 array

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
        
        # Initialize weights if needed
        if self.weights is None or self.weights.shape != (num_features, self.n_classes):
            # Random normal init
            rng = np.random.default_rng(self.random_state)
            self.weights = rng.normal(0, self.w_input, (num_features, self.n_classes)).astype(np.float32)
        
        # Initialize bias if needed
        if self.intercept and (self.b is None or self.b.shape != (self.n_classes,)):
            # Zero initialization for bias
            self.b = np.zeros(self.n_classes, dtype=np.float32)
        
        # Data label one-hot transform
        y_one_hot = one_hot_labeling(y_processed, self.classes)
        
        # Initialize Adam/AdamW moments if needed
        if self.optimizer in {'adam', 'adamw'}:
            # First moment for weights
            self.m_w = np.zeros_like(self.weights, dtype=np.float32)
            # Second moment for weights
            self.v_w = np.zeros_like(self.weights, dtype=np.float32)
            # First moment for bias
            self.m_b = np.zeros_like(self.b, dtype=np.float32) if self.intercept else None
            # Second moment for bias
            self.v_b = np.zeros_like(self.b, dtype=np.float32) if self.intercept else None
        
        # Random number generator for shuffling
        rng = np.random.default_rng(self.random_state)
        # Calculate number of batches
        num_batches = np.int32(np.ceil(num_samples / self.batch_size))
        # Initialize current learning rate
        self.current_lr = self.learning_rate
        # Initialize wait counter for early stopping
        self.wait = 0
        
        # ========== TRAINING LOOP ==========
        for i in range(self.max_iter):
            i = np.int32(i)
            # ========== LEARNING RATE SCHEDULING ==========
            # Apply LR scheduler after warm-up iterations
            if i > self.stoic_iter:
                if self.lr_scheduler == 'constant':
                    # Keep learning rate constant
                    self.current_lr = self.learning_rate

                elif self.lr_scheduler == 'invscaling':
                    # Inverse scaling decay
                    self.current_lr = self.current_lr / ((i + np.int32(1))**self.power_t + self.epsilon)
                
                elif self.lr_scheduler == 'adaptive':
                    # Adaptive learning rate based on loss ratio
                    ratio = np.sqrt(abs(np.mean(self.loss_history[-self.window:], dtype=np.float32) / np.mean(self.loss_history[-2 * self.window:-self.window], dtype=np.float32)))
                    if ratio <= 1:
                        self.current_lr = np.clip(self.current_lr / (i + 1)**self.power_t, self.epsilon, 10.0, dtype=np.float32)

                    else:
                        self.current_lr = np.clip(self.current_lr * np.sqrt(ratio), self.epsilon, 10.0, dtype=np.float32)

                elif self.lr_scheduler == 'plateau':
                    # Compute full dataset loss
                    current_loss = self._calculate_loss(y_one_hot, self.predict_proba(X_processed))
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

            # ========== DATA SHUFFLING ==========
            if self.shuffle:
                # Generate random permutation indices
                indices = rng.permutation(num_samples)
                # Shuffle X
                X_shuffled = X_processed[indices] if not issparse(X_processed) else X_processed[indices]
                # Shuffle y
                y_shuffled = y_one_hot[indices]
            # No shuffling
            else:
                X_shuffled = X_processed
                y_shuffled = y_one_hot

            # ========== BATCH PROCESSING ==========
            epoch_loss_sum = np.float32(0.0)
            for j in range(num_batches):
                j = np.int32(j)
                # Start index for current batch
                s_idx = j * self.batch_size
                # End index for current batch
                e_idx = min((j + np.int32(1)) * self.batch_size, num_samples)
                # Extract batch features
                X_batch = X_shuffled[s_idx:e_idx]
                # Extract batch labels
                y_batch = y_shuffled[s_idx:e_idx]
                # Compute gradients for batch
                grad_w, grad_b, z_batch = self._calculate_grad(X_batch, y_batch)

                # ========== PARAMETER UPDATES ==========
                if self.optimizer == 'mbgd':
                    # Update weights using MBGD
                    self.weights -= self.current_lr * grad_w
                    if self.intercept:
                        # Update bias using MBGD
                        self.b -= self.current_lr * grad_b

                elif self.optimizer == 'adam':
                    # Time step for Adam
                    t = i * num_batches + j + np.int32(1)
                    # Update first moment for weights
                    self.m_w = self.beta1 * self.m_w + (np.int32(1) - self.beta1) * grad_w
                    # Update second moment for weights
                    self.v_w = self.beta2 * self.v_w + (np.int32(1) - self.beta2) * (grad_w**np.int32(2))
                    # Bias-corrected first moment
                    m_w_hat = self.m_w / (np.int32(1) - self.beta1**t)
                    # Bias-corrected second moment
                    v_w_hat = self.v_w / (np.int32(1) - self.beta2**t)
                    # Update weights
                    self.weights -= self.current_lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

                    if self.intercept:
                        # Update first moment for bias
                        self.m_b = self.beta1 * self.m_b + (np.int32(1) - self.beta1) * grad_b
                        # Update second moment for bias
                        self.v_b = self.beta2 * self.v_b + (np.int32(1) - self.beta2) * (grad_b**np.int32(2))
                        # Bias-corrected first moment
                        m_b_hat = self.m_b / (np.int32(1) - self.beta1**t)
                        # Bias-corrected second moment
                        v_b_hat = self.v_b / (np.int32(1) - self.beta2**t)
                        # Update bias
                        self.b -= self.current_lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
                elif self.optimizer == 'adamw':
                    # Time step for AdamW
                    t = i * num_batches + j + np.int32(1)
                    # Update first moment for weights
                    self.m_w = self.beta1 * self.m_w + (np.int32(1) - self.beta1) * grad_w
                    # Update second moment for weights
                    self.v_w = self.beta2 * self.v_w + (np.int32(1) - self.beta2) * (grad_w**np.int32(2))
                    # Bias-corrected first moment
                    m_w_hat = self.m_w / (np.int32(1) - self.beta1**t)
                    # Bias-corrected second moment
                    v_w_hat = self.v_w / (np.int32(1) - self.beta2**t)
                    # Update weights
                    self.weights -= self.current_lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

                    if self.penalty == "l2":
                        # Apply L2 weight decay
                        self.weights -= self.current_lr * self.alpha * self.weights

                    if self.intercept:
                        # Update first moment for bias
                        self.m_b = self.beta1 * self.m_b + (np.int32(1) - self.beta1) * grad_b
                        # Update second moment for bias
                        self.v_b = self.beta2 * self.v_b + (np.int32(1) - self.beta2) * (grad_b**np.int32(2))
                        # Bias-corrected first moment
                        m_b_hat = self.m_b / (np.int32(1) - self.beta1**t)
                        # Bias-corrected second moment
                        v_b_hat = self.v_b / (np.int32(1) - self.beta2**t)
                        # Update bias
                        self.b -= self.current_lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

                # ========== BATCH LOSS COMPUTATION ==========
                # Compute softmax probabilities
                y_proba_batch = forlinear.softmax(z_batch)
                # Accumulate batch loss
                epoch_loss_sum += self._calculate_loss(y_batch, y_proba_batch)

            # ========== EPOCH LOSS AND LOGGING ==========
            # Average loss over all batches
            avg_epoch_loss = safe_array(epoch_loss_sum / num_batches)
                
            # Store epoch loss
            self.loss_history.append(avg_epoch_loss)

            # Check for NaN/Inf during training loop
            if not np.all(np.isfinite(self.weights)) or (self.intercept and not np.all(np.isfinite(self.b))):
                self.weights = safe_array(self.weights)
                
                if self.intercept:
                    self.b = safe_array(self.b)

            # Check loss for NaN/Inf during training loop
            if not np.isfinite(avg_epoch_loss):
                raise OverflowError(f"Loss became NaN/Inf at epoch {i + 1}. Stopping training early.")

            # Light verbose logging
            if self.verbose == 1 and ((i % max(1, self.max_iter // 20)) == 0 or i < 5) and self.verbosity == 'light':
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {avg_epoch_loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {np.mean(self.b):.6f}")

            elif self.verbose == 2 and self.verbosity == 'light':
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {avg_epoch_loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {np.mean(self.b):.6f}")

            # Heavy verbose logging
            if self.verbose == 1 and ((i % max(1, self.max_iter // 20)) == 0 or i < 5) and self.verbosity == 'heavy':
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {avg_epoch_loss:.8f}, Avg Weights: {np.mean(self.weights):.8f}, Avg Bias: {np.mean(self.b):.8f}, Current LR: {self.current_lr:.8f}")

            elif self.verbose == 2 and self.verbosity == 'heavy':
                print(f"Epoch {i + 1}/{self.max_iter}. Loss: {avg_epoch_loss:.8f}, Avg Weights: {np.mean(self.weights):.8f}, Avg Bias: {np.mean(self.b):.8f}, Current LR: {self.current_lr:.8f}")

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
            **ValueError**: *If model is not trained (propagated from predict_proba).*
        """
        # Get predicted probabilities
        probas = self.predict_proba(X_test)
        # Choose class with highest probability
        pred_class = np.argmax(probas, axis=1)

        if self.classes is not None and len(self.classes) == self.n_classes:
            # Map indices to original classes
            pred_class = np.array([self.classes[idx] for idx in pred_class], dtype=np.int32)
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
            "fit_intercept": self.intercept,
            "tol": self.tol,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "early_stopping": self.early_stop,
            "verbose": self.verbose,
            "verbosity": self.verbosity,
            "lr_scheduler": self.lr_scheduler,
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "power_t": self.power_t,
            "patience": self.patience,
            "factor": self.factor,
            "stoic_iter": self.stoic_iter,
            "epsilon": self.epsilon,
            "adalr_window": self.window,
            "start_w_scale": self.w_input
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