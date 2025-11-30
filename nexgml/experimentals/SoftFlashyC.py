# ========== LIBRARIES ==========
import numpy as np                           # For numerical operations
from scipy.sparse import issparse, spmatrix  # For sparse matrix handling
from typing import Literal, Optional         # More specific type hints
from nexgml.amo import softmax               # For some math operation
from nexgml.indexing import standard_indexing, one_hot_labeling  # For indexing utilities

# ========== THE MODEL ==========
class SoftFlashyClassifier:
    def __init__(
            self,
            max_iter: int=1000,
            learning_rate: float=0.01,
            verbose: int=0,
            fit_intercept: bool=True,
            tol: float=0.0001,
            penalty: Literal['l1', 'l2', 'elasticnet'] | None = 'l2',
            alpha: float = 0.0001,
            l1_ratio: float = 0.5,
            shuffle: bool=True,
            adaptive_alpha: bool | None=False,
            random_state: int | None=None,
            early_stopping: bool | None=True,
            max_features: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
            max_samples: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
            lr_scheduler: Literal['constant', 'invscaling'] | None='constant',
            power_t: float | None=0.25,
            stoic_iter: Optional[Literal['sqrt', 'log2']] | int | float | None=10
            ):
        """
        SoftFlashyC or SoftFlashyClassifier is a linear classifier that uses gradient descent optimization with softmax for multi-class classification.
        It supports L1, L2, and Elastic Net regularization to prevent overfitting.
        Categorical cross-entropy loss is used with class balancing via sample weights.
        """

        # ========== HYPERPARAMETERS ==========
        self.max_iter = int(max_iter)              # Model max training iterations
        self.lr_rate = float(learning_rate)        # Learning rate for gradient descent
        self.intercept = bool(fit_intercept)       # Fit intercept (bias) or not
        self.verbose = int(verbose)                # Model progress logging
        self.tol = float(tol)                      # Training loss tolerance for early stopping
        self.penalty = penalty                     # Penalties for regularization
        self.alpha = float(alpha)                  # Alpha for regularization power
        self.l1_ratio = float(l1_ratio)            # Elastic net mixing ratio
        self.shuffle = bool(shuffle)               # Data shuffling
        self.random_state = random_state           # For reproducibility
        self.early_stop = bool(early_stopping)     # Early stopping flag
        self.max_features = max_features           # Max features per iter
        self.max_samples = max_samples             # Max samples per iter
        self.lr_scheduler = lr_scheduler           # Learning rate schedule during training
        self.pow_t = power_t                       # Invscaling lr_scheduler power
        self.stoic_iter = stoic_iter               # Warm-up iteration
        self.ada_alpha = bool(adaptive_alpha)      # Adaptive alpha flag

        self.weights = None                        # Model weights
        self.b = None                              # Model bias
        self.loss_history = []                     # Store loss per-iteration

        self.classes = None                        # Unique classes
        self.n_classes = 0                         # Number of classes
        self.feature_indices = []                  # Store feature indices from data

    # ========= HELPER METHODS =========
    def cce(self, y_true, y_pred_proba, sample_weights=None):
        """
        Compute categorical cross-entropy loss with regularization.
        Supports sample weights for class balancing.
        L1, L2, and Elastic Net regularization available.
        """
        epsilon = 1e-15                                                                          # Small value to avoid log(0)
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)                               # Clip probabilities to avoid numerical issues

        if sample_weights is not None:
            ce = -np.sum(y_true * np.log(y_pred_proba) * sample_weights[:, np.newaxis], axis=1)          # Weighted sparse cross-entropy
            ce = np.mean(ce)                                                                     # Mean weighted CE
        else:
            ce = -np.mean(np.sum(y_true * np.log(y_pred_proba), axis=1))                  # Unweighted sparse cross-entropy

        reg = 0.0                                                                                # Regularization term initialization
        if self.weights is not None:
            if self.penalty == 'l1':
                reg = self.alpha * np.sum(np.abs(self.weights[self.feature_indices[-1]]))                                  # L1 regularization
            elif self.penalty == 'l2':
                reg = (self.alpha / 2) * np.sum(self.weights[self.feature_indices[-1]] ** 2)                               # L2 regularization
            elif self.penalty == 'elasticnet':
                l1 = self.alpha * self.l1_ratio * np.sum(np.abs(self.weights[self.feature_indices[-1]]))                   # L1 part of Elastic Net
                l2 = (1 - self.l1_ratio) * (self.alpha / 2) * np.sum(self.weights[self.feature_indices[-1]] ** 2)          # L2 part of Elastic Net
                reg = l1 + l2                                                                    # Total Elastic Net regularization
            else:
                raise ValueError(f"Invalid penalty argument {self.penalty}")

        return ce + reg                                                                          # Total loss with regularization

    def grad(self, X_scaled: np.ndarray | spmatrix, y_true: np.ndarray, sample_weights=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients of loss function with regularization.
        L1, L2, and Elastic Net available.
        """
        if not issparse(X_scaled):                                        # Ensure at least 2D for dense matrices
            X_scaled = np.atleast_2d(X_scaled)

        z = X_scaled @ self.weights[self.feature_indices[-1]]                                       # Linear combination
        if self.intercept:
            z += self.b                                                   # Add bias if intercept

        y_pred_proba = softmax(z)                                         # Compute softmax probabilities

        error = y_pred_proba - y_true                                     # Prediction error

        if sample_weights is not None:
            grad_w = (X_scaled.T @ (error * sample_weights[:, np.newaxis])) / X_scaled.shape[0]                         # Weighted weight gradient
            grad_b = np.average(error, weights=sample_weights, axis=0) if self.intercept else np.zeros(self.n_classes)  # Weighted bias gradient
        else:
            grad_w = (X_scaled.T @ error) / X_scaled.shape[0]             # Unweighted weight gradient
            grad_b = np.mean(error, axis=0) if self.intercept else np.zeros(self.n_classes) # Unweighted bias gradient

        if self.penalty == 'l1':
            grad_w += self.alpha * np.sign(self.weights[self.feature_indices[-1]])                  # L1 regularization gradient
        elif self.penalty == 'l2':
            grad_w += self.alpha * self.weights[self.feature_indices[-1]]                           # L2 regularization gradient
        elif self.penalty == 'elasticnet':
            l1_grad = self.alpha * self.l1_ratio * np.sign(self.weights[self.feature_indices[-1]])  # L1 part of Elastic Net
            l2_grad = (1 - self.l1_ratio) * self.alpha * self.weights[self.feature_indices[-1]]     # L2 part of Elastic Net
            grad_w += l1_grad + l2_grad                                   # Total Elastic Net gradient
        else:
            raise ValueError(f"Invalid penalty argument {self.penalty}")     # Warn for invalid penalty

        return grad_w, grad_b                                             # Return gradients for weights and bias

    def fit(self, X_train, y_train):
        """
        Fit the model to the training data using gradient descent method.
        """
        stoic_iter = standard_indexing(self.max_iter, self.stoic_iter)           # Determine stoic iter

        # Data check-up
        if not issparse(X_train):                            # Check if not sparse
            if X_train.ndim == 1:                            # Reshape 1D to 2D
                X_processed = X_train.reshape(-1, 1)
            else:
                X_processed = X_train
            X_processed = np.asarray(X_processed)            # Convert to numpy array
        else:
            X_processed = X_train                            # Keep sparse

        num_samples, num_features = X_processed.shape        # Get dimensions

        y_processed = np.asarray(y_train).ravel()            # Ensure y is 1D array

        if issparse(X_processed):                            # Check sparse data for NaN/Inf
            if not np.all(np.isfinite(X_processed.data)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values in its data. Please clean your data.")
        else:                                                # Check dense data for NaN/Inf
            if not np.all(np.isfinite(X_processed)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")

        if not np.all(np.isfinite(y_processed)):             # Check y for NaN/Inf
            raise ValueError("Input target (y_train) contains NaN or Infinity values. Please clean your data.")

        if X_processed.shape[0] != y_processed.shape[0]:     # Check sample count match
            raise ValueError(
                f"Number of samples in X_train ({X_processed.shape[0]}) "
                f"must match number of samples in y_train ({y_processed.shape[0]})."
            )

        self.classes = np.unique(y_processed)               # Unique classes
        self.n_classes = len(self.classes)                  # Number of classes
        self.loss_history = []                              # Initialize loss history

        if self.n_classes < 2:                              # Check for at least 2 classes
           raise ValueError("Class label must have at least 2 types.")

        y_onehot = one_hot_labeling(y_processed, self.classes)                   # Get sparse labels

        # Pre-train process
        if self.weights is None or self.weights.shape != (num_features, self.n_classes):  # Initialize weights if needed
            rng = np.random.default_rng(self.random_state)
            self.weights = rng.normal(0, 0.01, (num_features, self.n_classes))            # Random normal init

        self.b = np.zeros(self.n_classes)                   # Initialize bias

        class_counts = np.bincount(y_processed, minlength=self.n_classes)                 # Class counts
        total_samples = num_samples                         # Total samples
        class_weights = total_samples / (self.n_classes * class_counts + 1e-7)            # Class weights for balancing
        sample_weights = np.zeros(num_samples)              # Sample weights array
        for i, cls in enumerate(self.classes):
            sample_weights[y_processed == cls] = class_weights[i]                         # Assign sample weights

        max_features = standard_indexing(num_features, self.max_features)        # Determine max features
        max_samples = standard_indexing(num_samples, self.max_samples)           # Determine max samples
        
        # Training process
        for i in range(self.max_iter):                  # Iteration loop
            if i > stoic_iter:
                if self.lr_scheduler == 'invscaling':
                    self.lr_rate = self.lr_rate / ((i + 1)**self.pow_t + 1e-8)  # Update learning rate

            if self.shuffle:                            # Shuffle data if enabled
                indices = np.random.choice(num_samples, max_samples, replace=True)
                X_shuffled = X_processed[indices]       # Shuffle X
                y_onehot_shuffled = y_onehot[indices]   # Shuffle y
                sw_shuffled = sample_weights[indices]   # Shuffle sample weights
            else:                                       # No shuffle
                indices = np.random.choice(num_samples, max_samples, replace=False)
                X_shuffled = X_processed[indices]
                y_onehot_shuffled = y_onehot[indices]
                sw_shuffled = sample_weights[indices]

            feature_idx = np.random.choice(num_features, max_features, replace=False)
            self.feature_indices.append(feature_idx)
            X_feat = X_shuffled[:, feature_idx]

                                                        # Compute current predictions for loss and gradients
            z_current = X_feat @ self.weights[self.feature_indices[-1]]       # Linear combo
            if self.intercept:
                z_current += self.b                     # Add bias
            y_proba_current = softmax(z_current)    # Softmax probabilities

            # Compute loss using current proba
            loss = self.cce(y_onehot_shuffled, y_proba_current, sw_shuffled)        # Current loss
            self.loss_history.append(loss)              # Store loss

            if self.ada_alpha and i > stoic_iter:
                self.alpha *= (self.loss_history[-1] / self.loss_history[-2])

            # Compute gradients using current proba
            grad_w, grad_b = self.grad(X_feat, y_onehot_shuffled, sw_shuffled)  # Gradients

            # Update parameters
            self.weights[self.feature_indices[-1]] -= self.lr_rate * grad_w      # Update weights
            if self.intercept:
                self.b -= self.lr_rate * grad_b        # Update bias

            if not np.all(np.isfinite(self.weights)) or (self.intercept and not np.all(np.isfinite(self.b))):  # Check for NaN/Inf
                raise ValueError(f"Weights or bias became NaN/Inf at epoch {i + 1}. Stopping training early.")

            if not np.isfinite(loss):                  # Check loss for NaN/Inf
                raise ValueError(f"Loss became NaN/Inf at epoch {i + 1}. Stopping training early.")

            if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol and self.early_stop and i > stoic_iter:  # Early stopping
                break

            if self.verbose == 1:                      # Verbose logging
                print(f"Epoch {i+1}/{self.max_iter}, Grad_w: {np.mean(self.weights):.6f}, Grad_b: {np.mean(self.b):.6f} Loss: {loss:.6f}, Lr_rate: {self.lr_rate:.6f}, Alpha: {self.alpha:.6f}")

    def predict_proba(self, X_test: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict class probabilities using the trained model.
        """
        if not issparse(X_test):                                     # Check if not sparse
            if X_test.ndim == 1:                                     # Reshape 1D to 2D
                X_processed = X_test.reshape(-1, 1)
            else:
                X_processed = X_test
            X_processed = np.asarray(X_processed, dtype=np.float64)  # Convert to float array
        else:
            X_processed = X_test                                     # Keep sparse

        if self.n_classes == 0:                                      # Check if model is trained
             raise ValueError("Model not trained. Call fit() first.")

        if self.weights is None:                                     # Check if weights initialized
            raise ValueError("Weights not initialized. Call fit() first.")

        z = X_processed @ self.weights                               # Linear combination

        if self.intercept:                                           # Add bias if enabled
           z += self.b

        return softmax(z)                                        # Return softmax probabilities

    def predict(self, X_test: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict class labels using the trained model.
        """
        probas = self.predict_proba(X_test)                                   # Get probabilities

        pred_class = np.argmax(probas, axis=1)                                # Get class indices

        if self.classes is not None and len(self.classes) == self.n_classes:  # Map to original class labels
            pred_class = np.array([self.classes[idx] for idx in pred_class])

        return pred_class                                                     # Return predictions