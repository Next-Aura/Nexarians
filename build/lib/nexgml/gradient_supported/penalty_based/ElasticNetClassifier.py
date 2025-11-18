# ========== LIBRARIES ==========
import numpy as np                           # For numerical computations
from scipy.sparse import issparse, spmatrix  # For sparse data handling
import pandas as pd                          # For DataFrame data handling
from nexgml.indexing import one_hot_labeling # For encoding utility

# ========== THE MODEL ==========
class ElasticNetClassifier:
    """
    ElasticNet Classifier is a linear classifier that uses coordinate descent to minimize
    a loss function with ElasticNet (L1 + L2) regularization.
    It is suitable for multi-class classification by fitting a single multi-class model.
    """
    def __init__(self, 
                 max_iter: int=100,
                 alpha: float=1e-4, 
                 l1_ratio: float=0.5,
                 fit_intercept: bool=True, 
                 tol: float=1e-4,
                 early_stopping: bool=True,
                 verbose: int=0) -> None:
        """
        Initialize the ElasticNetClassifier model.

        ## Args:
            **max_iter**: *int, default=100*
            Maximum number of coordinate descent iterations.

            **alpha**: *float, default=1e-4*
            Regularization strength (used if penalty is not None).

            **l1_ratio**: *float, default=0.5*
            Mixing parameter for elastic net (0 <= l1_ratio <= 1).
            l1_ratio=0 is L2 (Ridge), l1_ratio=1 is L1 (Lasso).

            **fit_intercept**: *bool, default=True*
            If True, include a bias term (intercept).

            **tol**: *float, default=1e-4* Tolerance for convergence. The iterations will stop when the
            mean absolute change in weights is less than tol.

            **early_stopping**: *bool, default=True*
            If true, will make the model end the training loop early if the model in plateau performance.

            **verbose**: *int, default=0*
            If 1 or 2, print training progress (epoch, residual).

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
        self.l1_ratio = float(l1_ratio)            # Elastic net mixing ratio
        self.early_stop = bool(early_stopping)     # Early stopping flag

        self.weights = None                        # Model weights
        self.b = None                              # Model bias
        self.classes = None                        # Array of unique class labels from training data
        self.n_classes = None                      # Number of unique classes (determined during fit)
        self.loss_history = []                 # Store residual history per epoch

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
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])

    def _soft_thresholding(self, rho: np.ndarray, lam: float) -> np.ndarray:
        """
        Apply the soft-thresholding operation (proximal operator for L1).
        
        ## Args:
            **rho**: *np.ndarray*
            The values to threshold.

            **lam**: *float*
            The threshold (lambda).

        ## Returns:
            **np.ndarray**: *The thresholded values.*
        
        ## Raises:
            **None**
        """
        return np.sign(rho) * np.maximum(np.abs(rho) - lam, 0)

    def fit(self, X_train: np.ndarray | pd.DataFrame | spmatrix, y_train: np.ndarray | pd.Series) -> 'ElasticNetClassifier':
        """
        Fit the model to the training data using the Coordinate Descent algorithm.
        
        ## Args:
            **X_train**: *np.ndarray, pd.DataFrame, or sparse matrix*
            Training input features.

            **y_train**: *np.ndarray or pd.Series*
            Training target values (class labels).
            
        ## Returns:
            **ElasticNetClassifier**: *The fitted instance of the model.*
            
        ## Raises:
            **ValueError**: *If input data contains NaN/Inf, if X is not 2D, or if dimensions mismatch.*
        """
        # ========== Data Validation and Preprocessing ==========
        if issparse(X_train):
            if not np.all(np.isfinite(X_train.data)) or not np.all(np.isfinite(y_train)):
                raise ValueError(f"There's a NaN or infinity value between X and Y data, please clean your data first.")
            
        else:
            if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
                raise ValueError(f"There's a NaN or infinity value between X and Y data, please clean your data first.")

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
        assert n_samples == n_samples_y, "Mismatched number of samples between X and Y"

        if self.intercept: # Changed from self.fit_intercept
            X_aug = self._add_intercept(X)
        else:
            X_aug = X

        n_features_aug = X_aug.shape[1]

        # ========== Model Fitting (Coordinate Descent) ==========
        W = np.zeros((n_features_aug, n_classes), dtype=np.float64)

        # Precompute squared norms of features
        X_j_squared = np.sum(X_aug ** 2, axis=0)

        # Initialize residuals
        residual = Y.copy()

        for iteration in range(self.max_iter):
            W_old = W.copy()

            for j in range(n_features_aug):
                Xj = X_aug[:, j]

                # Calculate rho = X_j^T @ residual
                rho = np.dot(Xj.T, residual)

                if j == 0 and self.intercept:
                    # Intercept update (no regularization)
                    W[j, :] = rho / X_j_squared[j]
                else:
                    # ElasticNet update
                    # L2 penalty part
                    z = X_j_squared[j] + self.alpha * (1 - self.l1_ratio)
                    # L1 penalty part (soft thresholding)
                    W[j, :] = self._soft_thresholding(rho, self.alpha * self.l1_ratio) / z

                # Update residuals: r = r - X_j * (W_new[j] - W_old[j])
                delta = W[j, :] - W_old[j, :]
                residual -= np.outer(Xj, delta)

            residual_mean = np.mean(residual)
            self.loss_history.append(residual_mean)

            # Level 1 verbose logging
            if self.verbose == 1 and ((iteration % max(1, self.max_iter // 20)) == 0 or iteration < 5):
                print(f"Epoch {iteration + 1}/{self.max_iter}. Residual: {residual_mean:.6f}")

            # Level 2 verbose logging
            elif self.verbose == 2:
                print(f"Epoch {iteration + 1}/{self.max_iter}. Residual: {residual_mean:.6f}")

            # Check convergence
            if abs(np.mean(W - W_old)) < self.tol and self.early_stop:
                break

        # ========== Store Weights and Bias ==========
        if self.intercept: # Changed from self.fit_intercept
            self.b = W[0, :].reshape(1, -1)     # Changed from self.intercept_
            self.weights = W[1:, :]             # Changed from self.coef_
        else:
            self.b = np.zeros((1, n_classes), dtype=np.float64) # Changed from self.intercept_
            self.weights = W                    # Changed from self.coef_

        return self

    def predict(self, X_test: np.ndarray | pd.DataFrame | spmatrix) -> np.ndarray:
        """
        Predict class labels using the trained model.

        ## Args:
            **X_test**: *np.ndarray, pd.DataFrame, or spmatrix*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *Predicted class labels.*

        ## Raises:
            **ValueError**: *If the model is not fitted or if the number of features mismatches.*
        """
        # ========== Data Validation and Preprocessing ==========
        if issparse(X_test):
            if not np.all(np.isfinite(X_test.data)):
                raise ValueError(f"There's a NaN or infinity value in X_test data, please clean your data first.")
            
        else:
            if not np.all(np.isfinite(X_test)):
                raise ValueError(f"There's a NaN or infinity value in X_test data, please clean your data first.")

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
        
        # ========== Prediction ==========
        assert self.weights is not None, "Model not fitted" # Changed from self.coef_
        assert n_features == self.weights.shape[0], f"Feature mismatch: got {n_features}, expected {self.weights.shape[0]}" # Changed from self.coef_
        
        # Calculate raw scores (logits)
        preds = X @ self.weights + self.b # Changed from self.coef_ and self.intercept_
        
        # Choose class with highest score
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
            "l1_ratio": self.l1_ratio,
            "fit_intercept": self.intercept,
            "tol": self.tol,
            "early_stopping": self.early_stop,
            "verbose": self.verbose
        }

    def set_params(self, **params) -> 'ElasticNetClassifier':
        """
        Returns model's attribute that ready to set.

        ## Args:
            **params**: *dict*
            Model parameters to set.

        ## Returns:
            **ElasticNetClassifier**: *The model instance with updated parameters.*

        ## Raises:
            **None**
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self