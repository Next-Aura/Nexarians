# ========== LIBRARIES ==========
import numpy as np                                     # For numerical computations
from scipy.sparse import issparse, csr_matrix, hstack, spmatrix  # For sparse data handling
import pandas as pd                                    # For DataFrame data handling
from nexgml.indexing import one_hot_labeling           # For encoding utility

# ========== THE MODEL ==========
class L1Classifier:
    """
    L1 Classifier, also known as Lasso Regression, is a linear classifier that minimizes a loss function with L1 regularization.
    It uses coordinate descent for optimization, which is efficient for sparse solutions.
    This model is effectively a multi-class logistic regression with L1 penalty.
    """
    def __init__(self,
                 max_iter: int=100,
                 alpha: float=1e-4,
                 fit_intercept: bool=True,
                 tol: float=1e-4,
                 early_stopping: bool=True,
                 verbose: int=0):
        """
        Initialize the L1Classifier model.

        ## Args:
            **max_iter**: *int, default=100*
            Maximum number of iterations for coordinate descent.

            **alpha**: *float, default=1e-4*
            Regularization strength (lambda in the L1 regularization term). Must be non-negative.

            **fit_intercept**: *bool, default=True*
            If True, include a bias term (intercept) in the model.

            **tol**: *float, default=1e-4*
            Tolerance for stopping criteria.

            **early_stopping**: *bool, default=True*
            If true, will make the model end the training loop early if the model in plateau performance.

            **verbose**: *int, default=0*
            Verbosity level (0: no output, 1: some output, 2: full output).

        ## Returns:
            **None**

        ## Raises:
            **None**
        """
        # ========== HYPERPARAMETERS ===========
        self.alpha = float(alpha)                  # Alpha for regularization power
        self.intercept = bool(fit_intercept)       # Fit intercept (bias) or not
        self.verbose = int(verbose)                # Model progress logging
        self.max_iter = int(max_iter)              # Model max training iterations
        self.tol = float(tol)                      # Training loss tolerance
        self.early_stop = bool(early_stopping)     # Early stopping flag

        self.weights = None                        # Model weights
        self.b = None                              # Model bias
        self.classes = None                        # Array of unique class labels from training data
        self.n_classes = None                      # Number of unique classes (determined during fit)
        self.loss_history = []                     # Store residual history per epoch

    def _soft_threshold(self, z: np.ndarray, gamma: float) -> np.ndarray:
        """
        Soft thresholding operator for L1 regularization.

        ## Args:
            **z**: *np.ndarray*
            Input array.

            **gamma**: *float*
            Threshold value.

        ## Returns:
            **np.ndarray**: *Thresholded array.*

        ## Raises:
            **None**
        """
        return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Helper function to add a column of ones for the intercept term.

        ## Args:
            **X**: *np.ndarray*
            Input features array.

        ## Returns:
            **np.ndarray**: *Augmented array with an intercept column.*

        ## Raises:
            **None**
        """
        ones = csr_matrix(np.ones((X.shape[0], 1), dtype=X.dtype))
        return hstack([ones, X], format='csr')

    def fit(self, X_train: np.ndarray | pd.DataFrame | spmatrix, y_train: np.ndarray | pd.DataFrame) -> 'L1Classifier':
        """
        Fit the model to the training data using coordinate descent with L1 regularization.

        ## Args:
            **X_train**: *np.ndarray, pd.DataFrame, or sparse matrix*
            Training input features.

            **y_train**: *np.ndarray or pd.DataFrame*
            Training target values (class labels).

        ## Returns:
            **L1Classifier**: *The fitted instance of the model.*

        ## Raises:
            **ValueError**: *If input data contains NaN/Inf, if X is not 2D, or if dimensions mismatch.*
        """
        # ========== Data Validation and Preprocessing ==========
        if issparse(X_train):
            if not np.all(np.isfinite(X_train.data)) or not np.all(np.isfinite(y_train)):
                raise ValueError(f"There's a NaN or infinity value between X_train and y_train data, please clean your data first.")

        else:
            if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
                raise ValueError(f"There's a NaN or infinity value between X_train and y_train data, please clean your data first.")

        if isinstance(X_train, pd.DataFrame):
            X = X_train.to_numpy(dtype=np.float64)

        elif issparse(X_train):
            X = X_train.toarray().astype(np.float64)

        else:
            X = np.array(X_train, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X.shape}")

        X = np.asarray(X, dtype=np.float64)

        # ========== Label Processing ==========
        classes = np.unique(y_train)
        self.classes = classes
        self.n_classes = len(classes)
        y_one_hot = one_hot_labeling(y_train, classes)
        Y = np.asarray(y_one_hot, dtype=np.float64)
        n_samples, n_features = X.shape
        n_samples_y, n_classes = Y.shape
        assert n_samples == n_samples_y, "Error"

        is_sparse = issparse(X)

        # ========== Model Fitting (Coordinate Descent) ==========
        # Augment X with intercept column if needed
        if self.intercept:
            ones = csr_matrix(np.ones((n_samples, 1), dtype=np.float64)) if is_sparse else np.ones((n_samples, 1), dtype=np.float64)
            X_aug = hstack([ones, X], format='csr') if is_sparse else np.hstack([ones, X])
            n_features_aug = n_features + 1
        else:
            X_aug = X
            n_features_aug = n_features

        # Precompute column norms (X_j^T @ X_j) for efficiency
        norms = np.zeros(n_features_aug)
        for j in range(n_features_aug):
            Xj = X_aug[:, j]
            norms[j] = Xj.multiply(Xj).sum() if is_sparse else np.dot(Xj.T, Xj)
            # Handle potential zero norm (e.g., constant zero feature column)
            if norms[j] <= 1e-10:
                norms[j] = 1e-10  # Prevent division by zero, effectively ignoring this feature's update

        # Initialize weights and residuals
        w = np.zeros((n_features_aug, n_classes), dtype=np.float64)
        residual = Y.copy()  # Residuals: r = Y - X_aug @ w

        # Coordinate Descent Loop
        for iteration in range(self.max_iter):
            w_old = w.copy()

            for j in range(n_features_aug):
                Xj = X_aug[:, j]

                # Calculate rho = X_j^T @ r (correlation of feature j with residuals)
                if is_sparse:
                    rho = np.asarray((Xj.T @ residual)).ravel()
                else:
                    rho = np.dot(Xj.T, residual)

                # Calculate z = rho + w[j] * norms[j] (part of the numerator in the update rule)
                z = rho + w[j, :] * norms[j]

                # Determine the new coefficient w_new[j]
                if self.intercept and j == 0:  # Intercept coefficient (bias)
                    # No regularization for intercept
                    w_new_j = z / norms[j] if norms[j] > 1e-10 else w[j, :]
                else:  # Feature coefficients
                    # Apply soft thresholding for L1 regularization
                    w_new_j = self._soft_threshold(z, self.alpha)
                    # Normalize by the precomputed norm
                    w_new_j = w_new_j / norms[j] if norms[j] > 1e-10 else np.zeros_like(w_new_j)

                # Calculate the change in the coefficient
                delta = w_new_j - w[j, :]

                # Update the coefficient vector
                w[j, :] = w_new_j

                # Only update residuals if the coefficient changed significantly
                # This is the main efficiency improvement
                if np.any(np.abs(delta) > 1e-12):  # Use a small tolerance for numerical stability
                    if is_sparse:
                        xj_dense = Xj.toarray().ravel()
                    else:
                        xj_dense = Xj.ravel()
                    # Update residuals: r = r - X_j * (w_new[j] - w_old[j])
                    residual -= np.outer(xj_dense, delta)

            residual_mean = np.mean(residual)
            self.loss_history.append(residual_mean)

            # Level 1 verbose logging
            if self.verbose == 1 and ((iteration % max(1, self.max_iter // 20)) == 0 or iteration < 5):
                print(f"Epoch {iteration + 1}/{self.max_iter}. Residual: {residual_mean:.6f}")

            # Level 2 verbose logging
            elif self.verbose == 2:
                print(f"Epoch {iteration + 1}/{self.max_iter}. Residual: {residual_mean:.6f}")

            # Check for convergence based on change in coefficients
            if abs(np.mean(w - w_old)) < self.tol and self.early_stop:
                break

        # Assign final coefficients and intercept
        if self.intercept:
            self.b = w[0, :].squeeze()  # Squeeze if single output
            self.weights = w[1:, :]
        else:
            self.b = np.zeros(n_classes).squeeze()  # Squeeze if single output
            self.weights = w

        return self

    def predict(self, X_test: np.ndarray | pd.DataFrame | spmatrix) -> np.ndarray:
        """
        Predict class labels for the input data.

        ## Args:
            **X_test**: *np.ndarray, pd.DataFrame, or sparse matrix*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *Predicted class labels.*

        ## Raises:
            **ValueError**: *If input data contains NaN/Inf, if X is not 2D, or if model is not fitted.*
        """
        # ========== Data Validation and Preprocessing ==========
        if issparse(X_test):
            if not np.all(np.isfinite(X_test.data)):
                raise ValueError(f"There's a NaN or infinity value in X data, please clean your data first.")

        else:
            if not np.all(np.isfinite(X_test)):
                raise ValueError(f"There's a NaN or infinity value in X data, please clean your data first.")

        if isinstance(X_test, pd.DataFrame):
            X = X_test.to_numpy(dtype=np.float64)

        elif issparse(X_test):
            X = X_test.toarray().astype(np.float64)

        else:
            X = np.array(X_test, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X.shape}")

        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        assert self.weights is not None, "Model not fitted"
        assert n_features == self.weights.shape[0], f"Feature mismatch: got {n_features}, expected {self.weights.shape[0]}"

        # ========== Prediction ==========
        preds = X @ self.weights + self.b
        pred_class = np.argmax(preds, axis=1)
        # Map indices to original classes
        if self.classes is not None and len(self.classes) == self.n_classes:
            pred_class = np.array([self.classes[idx] for idx in pred_class])
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
            "alpha": self.alpha,
            "fit_intercept": self.intercept,
            "tol": self.tol,
            "early_stopping": self.early_stop,
            "verbose": self.verbose
        }

    def set_params(self, **params) -> 'L1Classifier':
        """
        Returns model's attribute that ready to set.

        ## Args:
            **params**: *dict*
            Model parameters to set.

        ## Returns:
            **L1Classifier**: *The model instance with updated parameters.*

        ## Raises:
            **None**
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self