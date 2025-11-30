import numpy as np

class Support:
    def __init__(self, 
                 alpha=1e-3, 
                 fit_intercept=True):
        
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.n_classes_ = None
        self.feature_indices_ = None

    def _add_intercept(self, X):
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n_samples, n_features = X.shape
        n_samples_y, n_classes = Y.shape
        assert n_samples == n_samples_y, "Error"

        self.n_features_in_ = n_features
        self.n_classes_ = n_classes

        if self.fit_intercept:
            X_aug = self._add_intercept(X) 
        else:
            X_aug = X

        XtX = X_aug.T @ X_aug
        d = XtX.shape[0]
        reg = np.eye(d) * self.alpha
        if self.fit_intercept:
            reg[0, 0] = 0.0
        A = XtX + reg
        Xty = X_aug.T @ Y

        try:
            W = np.linalg.solve(A, Xty)

        except np.linalg.LinAlgError:
            W = np.linalg.pinv(A) @ Xty

        if self.fit_intercept:
            self.intercept_ = W[0, :].reshape(1, -1)
            self.coef_ = W[1:, :]
        else:
            self.intercept_ = np.zeros((1, n_classes), dtype=np.float64)
            self.coef_ = W
        
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        assert self.coef_ is not None, "Model not fitted"
        assert n_features == self.coef_.shape[0], f"Feature mismatch: got {n_features}, expected {self.coef_.shape[0]}"
        preds = X @ self.coef_ + self.intercept_
        return preds