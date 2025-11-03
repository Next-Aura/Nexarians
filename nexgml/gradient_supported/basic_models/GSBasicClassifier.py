# ========== LIBRARIES ==========
import numpy as np                           # Numpy for numerical computations
from scipy.sparse import issparse, spmatrix  # For sparse matrix handling
from typing import Literal, Optional         # More specific type hints
from ...helper import AMO                    # For some math operation
from ...helper import Indexing               # For indexing utilities

# ========== THE MODEL ==========
class BasicClassifier:
    """
    Gradient Supported Basic Classifier (GSBC) is a linear classifier that uses gradient descent optimization with softmax for multi-class classification.
    It supports L1, L2, and Elastic Net regularization to prevent overfitting.
    Uses logistic regression with gradient descent to minimize categorical cross-entropy loss.
    Handle both dense and sparse input matrices.
    """
    def __init__(
            self,
            max_iter: int=1000,
            learning_rate: float=0.01,
            penalty: Optional[Literal['l1', 'l2', 'elasticnet']] | None = 'l2',
            alpha: float = 0.0001,
            l1_ratio: float = 0.5,
            fit_intercept: bool=True,
            tol: float=0.0001,
            shuffle: bool=True,
            random_state: int | None=None,
            early_stopping: bool | None=True,
            verbose: int=0,
            ):
        """
        Initialize the SoftBasicClassifier model.

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
            **ValueError**: *If invalid penalty type is provided.*
        """
        # ========== PARAMETER VALIDATION ==========
        if penalty not in (None, "l1", "l2", "elasticnet"):
            raise ValueError(f"Invalid penalty argument {penalty}.")

        # ========== HYPERPARAMETERS ==========
        self.max_iter = int(max_iter)              # Model max training iterations
        self.learning_rate = float(learning_rate)  # Learning rate for gradient descent
        self.intercept = bool(fit_intercept)       # Fit intercept (bias) or not
        self.verbose = int(verbose)                # Model progress logging
        self.tol = float(tol)                      # Training loss tolerance for early stopping
        self.penalty = penalty                     # Penalties for regularization
        self.alpha = float(alpha)                  # Alpha for regularization power
        self.l1_ratio = float(l1_ratio)            # Elastic net mixing ratio
        self.shuffle = bool(shuffle)               # Data shuffling
        self.random_state = random_state           # For reproducibility
        self.early_stop = bool(early_stopping)     # Early stopping flag

        self.weights = None                        # Model weights
        self.b = None                              # Model bias
        self.loss_history = []                     # Store loss per-iteration
        self.epsilon = 1e-15                       # For numerical stability

        self.classes = None                        # Unique classes
        self.n_classes = 0                         # Number of classes

    # ========= HELPER METHODS =========
    def _calculate_loss(self, y_true: np.ndarray | spmatrix, y_pred_proba: np.ndarray | spmatrix, sample_weights: np.ndarray | None=None) -> float:
        """
        Compute categorical cross-entropy loss with regularization.
        Supports sample weights for class balancing.
        L1, L2, and Elastic Net regularization available.

        ## Args:
            **y_true**: *True one-hot encoded labels.*
            **y_pred_proba**: *Predicted class probabilities.*
            **sample_weights**: *Optional sample weights for weighted loss.*

        ## Returns:
            **float**: *Computed loss value.*

        ## Raises:
            **None**
        """
        y_pred_proba = np.clip(y_pred_proba, self.epsilon, 1 - self.epsilon)
        # Clip probabilities to avoid numerical issues

        if sample_weights is not None:
            # Weighted cross-entropy
            ce = -np.sum(y_true * np.log(y_pred_proba) * sample_weights[:, np.newaxis], axis=1)
            # Mean weighted CE
            ce = np.mean(ce)
        else:
            # Unweighted cross-entropy
            ce = -np.mean(np.sum(y_true * np.log(y_pred_proba), axis=1))

        reg = 0.0                  # Regularization term initialization

        # L1 regularization
        if self.penalty == 'l1':
            reg = self.alpha * np.sum(np.abs(self.weights))
        
        # L2 regularization
        elif self.penalty == 'l2':
            reg = self.alpha * np.sum(self.weights ** 2)
        
        # Elastic Net regularization
        elif self.penalty == 'elasticnet':
            # L1 part of Elastic Net
            l1 = self.alpha * self.l1_ratio * np.sum(np.abs(self.weights))
            # L2 part of Elastic Net
            l2 = (1 - self.l1_ratio) * self.alpha * np.sum(self.weights**2)
            # Total Elastic Net regularization
            reg = l1 + l2
        
        # Loss after regularization
        return ce + reg                                                                          # Total loss with regularization

    def _calculate_grad(self, X: np.ndarray | spmatrix, y: np.ndarray, sample_weights: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate gradient of loss function with regulation.
        support sample weights for weighted gradient.
        L1, L2, and Elastic Net available.
        
        ## Args:
            **X**: *np.ndarray*
            Input features.

            **y**: *np.ndarray*
            True target values.

            **sample_weights**: *np.ndarray*
            Sample weights for weighted gradient.
            
        ## Return:
            **tuple**: *gradient w.r.t. weights, gradient w.r.t. bias*
            
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
        y_pred_proba = AMO.softmax(z)
        
        # Prediction error
        error = y_pred_proba - y

        if sample_weights is not None:
            # Weighted weight gradient
            grad_w = (X.T @ (error * sample_weights[:, np.newaxis])) / X.shape[0]
            # Weighted bias gradient
            grad_b = np.average(error, weights=sample_weights, axis=0) if self.intercept else np.zeros(self.n_classes)
        
        else:
            # Unweighted weight gradient
            grad_w = (X.T @ error) / X.shape[0]
            # Unweighted bias gradient
            grad_b = np.mean(error, axis=0) if self.intercept else np.zeros(self.n_classes)
        
        # L1 regularization gradient
        if self.penalty == 'l1':
            grad_w += self.alpha * np.sign(self.weights)

        # L2 regularization gradient
        elif self.penalty == 'l2':
            grad_w += 2 * self.alpha * self.weights
        
        # Elastic Net regularization
        elif self.penalty == 'elasticnet':
            # L1 part of Elastic Net
            l1_grad = self.l1_ratio * np.sign(self.weights)
            # L2 part of Elastic Net
            l2_grad = 2 * ((1 - self.l1_ratio) * self.weights)
            # Total Elastic Net gradient
            grad_w += self.alpha * (l1_grad + l2_grad)
        
        # Return gradients for weights and bias
        return grad_w, grad_b
    
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
        return AMO.softmax(z)

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
            **OverflowError**: *If parameters (weight, bias or loss) become infinity or NaN during training loop.*
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
        y_int = Indexing.integer_labeling(y_processed, self.classes, to_integer_from='labels')

        # Get one-hot from integer labels
        y_onehot = Indexing.one_hot_labeling(y_int, np.arange(self.n_classes))

        # ---------- Pre-train process ----------
        # Initialize weights if needed
        if self.weights is None or self.weights.shape != (num_features, self.n_classes):
            # Random normal init
            rng = np.random.default_rng(self.random_state)
            self.weights = rng.normal(0, 0.01, (num_features, self.n_classes))

        self.b = np.zeros(self.n_classes)                   # Initialize bias
        
        # Class counts (use integer labels mapped from original classes)
        class_counts = np.bincount(y_int, minlength=self.n_classes)
        # Total samples
        total_samples = num_samples
        # Class weights for balancing
        class_weights = total_samples / (self.n_classes * class_counts + self.epsilon)
        # Sample weights array
        sample_weights = np.zeros(num_samples)

        # Assign weights to samples based on class
        for i, cls in enumerate(self.classes):
            # Assign sample weights using integer labels
            sample_weights[y_int == i] = class_weights[i]

        # RNG for shuffling
        rng = np.random.default_rng(self.random_state)
        # Iteration loop
        for i in range(self.max_iter):
            # Shuffle data if enabled
            if self.shuffle:
                indices = rng.permutation(num_samples)  # Permutation indices
                X_processed = X_processed[indices]       # Shuffle X
                y_onehot = y_onehot[indices]   # Shuffle y
                sample_weights = sample_weights[indices]   # Shuffle sample weights

            # Linear combination
            z_current = X_processed @ self.weights
            if self.intercept:
                # Add bias if intercept is used
                z_current += self.b

            # Get logits probabilities with softmax
            y_proba_current = AMO.softmax(z_current)

            # Current loss
            loss = self._calculate_loss(y_onehot, y_proba_current, sample_weights)
            # Store loss
            self.loss_history.append(loss)

            # Compute gradients using current proba
            grad_w, grad_b = self._calculate_grad(X_processed, y_onehot, sample_weights)

            # Update weights
            self.weights -= self.learning_rate * grad_w

            if self.intercept:
                # Update bias if intercept if used
                self.b -= self.learning_rate * grad_b

            # Check for NaN/Inf during training loop
            if not np.all(np.isfinite(self.weights)) or (self.intercept and not np.all(np.isfinite(self.b))):
                raise OverflowError(f"Weights or bias became NaN/Inf at epoch {i + 1}. Stopping training early.")

            # Check loss for NaN/Inf during training loop
            if not np.isfinite(loss):
                raise OverflowError(f"Loss became NaN/Inf at epoch {i + 1}. Stopping training early.")

            # Early stopping
            if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol and self.early_stop:
                break

            # Verbose logging
            if self.verbose == 1 and ((i % max(1, self.max_iter // 20)) == 0 or i < 5):
                print(f"Epoch {i+1}/{self.max_iter}, Loss: {loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {np.mean(self.b):.6f}")

            elif self.verbose == 2:
                print(f"Epoch {i+1}/{self.max_iter}, Loss: {loss:.6f}, Avg Weights: {np.mean(self.weights):.6f}, Avg Bias: {np.mean(self.b):.6f}")


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