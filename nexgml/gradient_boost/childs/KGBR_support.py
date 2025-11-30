import numpy as np

class Support:
    def __init__(self,
                 alpha=1e-3,
                 l1_ratio=0.5,
                 fit_intercept=True,
                 max_iter=1000,
                 tol=1e-4):
        """
        ElasticNet regression support class using coordinate descent.

        Parameters:
        - alpha: float, regularization strength
        - l1_ratio: float, the ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
                    For l1_ratio = 0 the penalty is an L2 penalty.
                    For l1_ratio = 1 it is an L1 penalty.
        - fit_intercept: bool, whether to calculate the intercept for this model.
        - max_iter: int, maximum number of iterations for coordinate descent.
        - tol: float, tolerance for the optimization.
        - verbose: int, verbosity level.
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.n_classes_ = None
        self.feature_indices_ = None

    def _add_intercept(self, X):
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])

    def _soft_thresholding(self, rho, lam):
        if rho < -lam:
            return rho + lam
        elif rho > lam:
            return rho - lam
        else:
            return 0.0

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n_samples, n_features = X.shape
        n_samples_y, n_classes = Y.shape
        assert n_samples == n_samples_y, "Mismatched number of samples between X and Y"

        self.n_features_in_ = n_features
        self.n_classes_ = n_classes

        if self.fit_intercept:
            X_aug = self._add_intercept(X)
        else:
            X_aug = X

        d = X_aug.shape[1]
        self.coef_ = np.zeros((d, n_classes), dtype=np.float64)

        for k in range(n_classes):
            y = Y[:, k]
            w = np.zeros(d, dtype=np.float64)
            for iteration in range(self.max_iter):
                w_old = w.copy()
                for j in range(d):
                    if self.fit_intercept and j == 0:
                        # Update intercept without regularization
                        residual = y - (X_aug @ w) + w[j] * X_aug[:, j]
                        w[j] = np.sum(residual) / n_samples
                    else:
                        residual = y - (X_aug @ w) + w[j] * X_aug[:, j]
                        rho = np.dot(X_aug[:, j], residual)
                        z = np.sum(X_aug[:, j] ** 2)
                        lam1 = self.alpha * self.l1_ratio
                        lam2 = self.alpha * (1 - self.l1_ratio)
                        w_j = self._soft_thresholding(rho, lam1) / (z + lam2)
                        w[j] = w_j

                if np.max(np.abs(w - w_old)) < self.tol:
                    break

            self.coef_[:, k] = w

        if self.fit_intercept:
            self.intercept_ = self.coef_[0, :].reshape(1, -1)
            self.coef_ = self.coef_[1:, :]
        else:
            self.intercept_ = np.zeros((1, n_classes), dtype=np.float64)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        assert self.coef_ is not None, "Model not fitted"
        assert n_features == self.coef_.shape[0], f"Feature mismatch: got {n_features}, expected {self.coef_.shape[0]}"
        preds = X @ self.coef_ + self.intercept_
        return preds
