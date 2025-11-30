import numpy as np

class Support:
    def __init__(self, 
                 alpha=1e-3, 
                 l1_ratio=0.5,
                 fit_intercept=True, 
                 max_iter=1000,
                 tol=1e-4):
        """
        ElasticNet regression solver using coordinate descent.
        alpha: regularization strength
        l1_ratio: float between 0 and 1, the ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
                  For l1_ratio = 0 the penalty is an L2 penalty.
                  For l1_ratio = 1 it is an L1 penalty.
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

        n_features_aug = X_aug.shape[1]

        # Initialize weights
        W = np.zeros((n_features_aug, n_classes), dtype=np.float64)

        # Precompute some useful quantities
        X_j_squared = np.sum(X_aug ** 2, axis=0)

        for class_idx in range(n_classes):
            w = W[:, class_idx]
            y = Y[:, class_idx]

            for iteration in range(self.max_iter):
                w_old = w.copy()

                for j in range(n_features_aug):
                    residual = y - X_aug @ w + w[j] * X_aug[:, j]
                    rho = np.dot(X_aug[:, j], residual)

                    if j == 0 and self.fit_intercept:
                        # Intercept update (no regularization)
                        w[j] = rho / X_j_squared[j]
                    else:
                        # ElasticNet update
                        z = X_j_squared[j] + self.alpha * (1 - self.l1_ratio)
                        w_j = self._soft_thresholding(rho, self.alpha * self.l1_ratio) / z
                        w[j] = w_j

                # Check convergence
                if np.linalg.norm(w - w_old, ord=1) < self.tol:
                    break

            W[:, class_idx] = w

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
