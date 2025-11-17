# ========== LIBRARIES ==========
import numpy as np                                     # For numerical computations
from scipy.sparse import issparse, csr_matrix, hstack, spmatrix  # For sparse matrix handling
import pandas as pd                                    # For DataFrame data support

# ========== THE MODEL ==========
class L1Regressor:
    """
    L1 Regressor, also known as Lasso Regression, is a linear regression model that uses
    L1 regularization to prevent overfitting and perform feature selection. It finds the optimal weights using Coordinate Descent.
    """

    def __init__(self,
                 max_iter: int=100,
                 alpha: float=1e-4,
                 fit_intercept: bool=True,
                 tol: float=1e-4,
                 early_stopping: bool=True,
                 verbose: int=0):
        """
        Initialize the L1Regressor model.

        ## Args:
            **max_iter**: *int, default=100*
            Maximum number of iterations for Coordinate Descent.

            **alpha**: *float, default=1e-4*
            Regularization strength for L1 penalty.

            **fit_intercept**: *bool, default=True*
            If True, the model will calculate the intercept (bias) term.

            **tol**: *float, default=1e-4*
            Tolerance for convergence in Coordinate Descent.

            **early_stopping**: *bool, default=True*
            If true, will make the model end the training loop early if the model in plateau performance.

            **verbose**: *int, default=0*
            Verbosity level (0: no output, 1: some output, 2: detailed output).

        ## Returns:
          **None**

        ## Raises:
          **None**
        """
        # =========== HYPERPARAMETERS ==========
        self.alpha = float(alpha)                  # Alpha for regularization power
        self.intercept = bool(fit_intercept)       # Fit intercept (bias) or not
        self.verbose = int(verbose)                # Model progress logging
        self.max_iter = int(max_iter)              # Model max training iterations
        self.tol = float(tol)                      # Training loss tolerance
        self.early_stop = bool(early_stopping)     # Early stopping flag

        self.weights = None                        # Model weights
        self.b = None                              # Model bias
        self.n_outputs_ = None                     # Number of y data outputs
        self.loss_history = []                     # Store residual history per epoch

    def _soft_threshold(self, z: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply soft thresholding operator for L1 regularization.

        ## Args:
            **z**: *np.ndarray*
            Input array to threshold.

            **gamma**: *float*
            Threshold value (related to regularization strength).

        ## Returns:
            **np.ndarray**: *Soft-thresholded array.*

        ## Raises:
          **None**
        """
        return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)

    def fit(self, X_train: np.ndarray | spmatrix | pd.DataFrame, y_train: np.ndarray | pd.Series) -> 'L1Regressor':
        """
        Fit the L1 Regressor model to the training data using Coordinate Descent.

        ## Args:
            **X_train**: *np.ndarray, pd.DataFrame, or spmatrix*
            Training input features.

            **y_train**: *np.ndarray or pd.Series*
            Training target values.

        ## Returns:
            **L1Regressor**: *self.*
            The fitted model object.

        ## Raises:
            **ValueError**: *If input data contains NaN/Inf or if dimensions mismatch.*
        """
        # Handle pandas inputs and ensure numpy arrays for dense, keep sparse as sparse
        if 'pandas' in str(type(X_train)):
            X = X_train.to_numpy(dtype=np.float64)
        if 'pandas' in str(type(y_train)):
            Y = y_train.to_numpy(dtype=np.float64)
        
        # Convert non-sparse inputs to numpy array, keep sparse inputs as sparse
        if not issparse(X_train):
            X = np.asarray(X_train, dtype=np.float64)
        else:
            # Ensure sparse matrix is in a compatible dtype if necessary
            # Often not needed if input is already float64, but good practice
            if X_train.dtype != np.float64:
                 X = X_train.astype(np.float64, copy=False) # copy=False avoids copying if dtype is already correct

        Y = np.asarray(y_train, dtype=np.float64)

        # Handle single output
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]

        n_samples, n_outputs = Y.shape
        if X.ndim != 2 or X.shape[0] != n_samples:
            raise ValueError(f"Expected 2D array for X with {n_samples} samples, got shape {X.shape}")

        # Check for finite values
        if issparse(X):
            if not np.all(np.isfinite(X.data)) or not np.all(np.isfinite(Y)):
                raise ValueError("NaN or infinity in data")
        else:
            if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
                raise ValueError("NaN or infinity in data")

        self.n_outputs_ = n_outputs
        is_sparse = issparse(X)

        # Augment X with intercept column if needed
        if self.intercept:
            ones = csr_matrix(np.ones((n_samples, 1), dtype=np.float64)) if is_sparse else np.ones((n_samples, 1), dtype=np.float64)
            X_aug = hstack([ones, X], format='csr') if is_sparse else np.hstack([ones, X])
            n_features_aug = X.shape[1] + 1
        else:
            X_aug = X
            n_features_aug = X.shape[1]

        # Precompute column norms (X_j^T @ X_j) for efficiency
        norms = np.zeros(n_features_aug)
        for j in range(n_features_aug):
            Xj = X_aug[:, j]
            norms[j] = Xj.multiply(Xj).sum() if is_sparse else np.dot(Xj.T, Xj)


            # Handle potential zero norm (e.g., constant zero feature column)
            if norms[j] <= 1e-10:
                 norms[j] = 1e-10 # Prevent division by zero, effectively ignoring this feature's update

        # Initialize weights and residuals
        w = np.zeros((n_features_aug, n_outputs), dtype=np.float64)
        residual = Y.copy() # Residuals: r = Y - X_aug @ w

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
                if self.intercept and j == 0: # Intercept coefficient (bias)
                    # No regularization for intercept
                    w_new_j = z / norms[j] if norms[j] > 1e-10 else w[j, :]
                else: # Feature coefficients
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
                if np.any(np.abs(delta) > 1e-12): # Use a small tolerance for numerical stability
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
            # Squeeze if single output
            self.b = w[0, :].squeeze()
            self.weights = w[1:, :]
        else:
            # Squeeze if single output
            self.b = np.zeros(n_outputs).squeeze()
            self.weights = w[:, :]

        return self

    def predict(self, X_test: np.ndarray | pd.DataFrame | spmatrix) -> np.ndarray:
        """
        Predict target values using the fitted L1 Regressor model.

        ## Args:
            **X_test**: *np.ndarray, pd.DataFrame, or spmatrix*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *Predicted target values.*
            Shape is (n_samples,) for single output or (n_samples, n_outputs) for multi-output.

        ## Raises:
            **ValueError**: *If input data contains NaN/Inf or if dimensions mismatch.*
        """
        if issparse(X_test):
            if not np.all(np.isfinite(X_test.data)):
                raise ValueError(f"There's a NaN or infinity value between in X data, please clean your data first.")
            
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

        # Perform the prediction
        preds = X @ self.weights + self.b
        return preds.squeeze() if self.n_outputs_ == 1 else preds

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