import numpy as np  # Numpy for numerical computations

# ========== JIT INITIALIZATION ==========
try:
    from numba import jit

except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ========== AMO HELPER ==========
jit(nopython=True)
class AMO:
    """AMO (Advanced Math Operations) for simple machine learning computations"""
    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """
        Calculate the softmax probability of the given logits.

        ## Args:
          **z**: *np.ndarray*
          Raw logits.

        ## Returns:
          **np.ndarray**: *Probability of the given logits.*

        ## Raises:
          **None**
        """
        z = np.asarray(z)
        if z.ndim == 1:
            z = z.reshape(1, -1)
            squeeze = True

        else:
            squeeze = False

        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        exp_z_sum = np.sum(exp_z, axis=1, keepdims=True)
        exp_z_sum = np.where(exp_z_sum == 0, 1, exp_z_sum)
        out = exp_z / exp_z_sum

        return out[0] if squeeze else out
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Calculate the sigmoid probability of the given logits.

        ## Args:
          **z**: *np.ndarray*
          Raw logits.

        ## Returns:
          **np.ndarray**: *Probability of the given logits.*

        ## Raises:
          **None**
        """
        try:
            from scipy.special import expit
            return expit(z)
        
        except Exception:
            z_maxi = np.clip(z, -500, 500)
            return 1 / (1 + np.exp(-z_maxi))
    
    @staticmethod
    def categorical_ce(y_true: np.ndarray, y_pred_proba: np.ndarray, mean: bool=True) -> np.ndarray:
        """
        Calculate classification loss using categorical cross-entropy formula.

        ## Args:
          **y_true**: *np.ndarray*
          True labels data.

          **y_pred_proba**: *np.ndarray*
          Labels prediction probability.

          **mean**: *bool, default=True*
          Return loss mean or not.

        ## Returns:
          **np.ndarray**: *Labels prediction probability loss.*

        ## Raises:
          **None**
        """
        epsilon = 1e-8
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

        class_counts = np.sum(y_true, axis=0)
        n_classes = len(class_counts)
        total = np.sum(class_counts)
        class_weights = total / (n_classes * class_counts + 1e-8)

        if np.sum(class_weights) == 0:
            class_weights = np.ones_like(class_weights)

        else:
            class_weights = class_weights / np.sum(class_weights)

        loss = -np.sum(class_weights * y_true * np.log(y_pred_proba), axis=1)

        if mean:
            return np.mean(loss)
        
        else:
            return loss
    
    @staticmethod
    def binary_ce(y_true: np.ndarray, y_pred_proba: np.ndarray, mean: bool=True) -> np.ndarray:
        """
        Calculate classification loss using binary cross-entropy formula.

        ## Args:
          **y_true**: *np.ndarray*
          True labels data.

          **y_pred_proba**: *np.ndarray*
          Labels prediction probability.

          **mean**: *bool, default=True*
          Return loss mean or not.

        ## Returns:
          **np.ndarray**: *Labels prediction probability loss.*

        ## Raises:
          **None**
        """
        epsilon = 1e-8
        y_pred_clip = np.clip(y_pred_proba, epsilon, 1 - epsilon)

        loss = -(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))

        if mean:
            return np.mean(loss)
        
        else:
            return loss
        
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate regression loss using mean squared error (MSE) formula.

        ## Args:
          **y_true**: *np.ndarray*
          True target data.

          **y_pred**: *np.ndarray*
          Target prediction.

        ## Returns:
          **float**: *Target prediction loss.*

        ## Raises:
          **None**
        """
        return np.mean((y_true - y_pred)**2)
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate regression loss using mean absolute error (MAE) formula.

        ## Args:
          **y_true**: *np.ndarray*
          True target data.

          **y_pred**: *np.ndarray*
          Target prediction.

        ## Returns:
          **float**: *Target prediction loss.*

        ## Raises:
          **None**
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def root_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate regression loss using root mean squared error (RMSE) formula.

        ## Args:
          **y_true**: *np.ndarray*
          True target data.

          **y_pred**: *np.ndarray*
          Target prediction.

        ## Returns:
          **float**: *Target prediction loss.*

        ## Raises:
          **None**
        """
        return np.sqrt(np.mean((y_true - y_pred)**2))
    
    @staticmethod
    def smoothl1_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float) -> float:
        """
        Calculate regression loss using smooth L1 (huber) loss formula.

        ## Args:
          **y_true**: *np.ndarray*
          True target data.

          **y_pred**: *np.ndarray*
          Target prediction.
          
          **delta**: *float*
          Function threshold between operation

        ## Returns:
          **float**: *Target prediction loss.*

        ## Raises:
          **None**
        """
        diff = np.abs(y_true - y_pred)
        loss = np.where(diff < delta, 0.5 * diff**2 / delta, diff - 0.5 * delta)

        return np.mean(loss)
    
jit(nopython=True)
class ForTree:
        """Helper for tree models operations"""
        @staticmethod
        def squared_error(labels: np.ndarray) -> float:
            """
            Calculate the variance of the given labels (MSE).

            ## Args:
                **labels**: *np.ndarray*
                Array of target values.

            ## Returns:
                **float**: *Variance of the labels. Returns 0.0 if labels are empty.*
                
            ## Raises:
                **None**
            """
            # Check label's size
            if labels.size == 0:
                # If the size is 0, then return 0.0
                return 0.0
            
            # Calculate label mean
            mean = np.mean(labels)
            # Calculate label variance (MSE)
            return np.mean((labels - mean) ** 2)
        
        @staticmethod
        def friedman_squared_error(labels: np.ndarray) -> float:
            n = labels.size

            if n <= 1:
                return 0.0
            
            mean = np.mean((labels - np.mean(labels))**2)

            return mean * (n / (n - 1))
        
        @staticmethod
        def absolute_error(y: np.ndarray) -> float:
            """
            Calculate the mean absolute error of the given labels.

            ## Args:
                **y**: *np.ndarray*
                Array of target values.

            ## Returns:
                **float**: *Mean absolute error of the labels. Returns 0.0 if labels are empty.*
                
            ## Raises:
                **None**
            """
            # Check labels array size
            if y.size == 0:
                # If the size is 0, then return 0.0
                return 0.0
            
            # Calculate labels mean
            mean = np.mean(y)
            # Calculate label variance (absolute error)
            return np.mean(np.abs(y - mean))
        
        @staticmethod
        def poisson_deviance(y: np.ndarray) -> float:
            """
            Calculate the Poisson deviance of the given labels.

            ## Args:
                **y**: *np.ndarray*
                Array of target values.

            ## Returns:
                **float**: *Poisson deviance of the labels.*

            ## Raises:
                **ValueError**: *If target values are negative.*
            """
            # Check labels array size
            if y.size == 0:
                # If the size is 0, then return 0.0
                return 0.0
            
            # Check if there's no labels that less than 0
            if np.any(y < 0):
                # If it's exist throw an error
                raise ValueError("Poisson deviance requires non-negative target values.")
            
            # Calculate labels mean
            mean_y = np.mean(y)

            # If labels mean is less than 0, return 0.0
            if mean_y <= 0:
                return 0.0

            # Calculate labels variance (poisson deviance)
            return 2.0 * np.sum(y * np.log(np.maximum(y, 1e-9) / mean_y) - (y - mean_y))
        
        @staticmethod
        def gini_impurity(labels: np.ndarray) -> float:
            """
            Calculate the Gini impurity for a set of labels.

            Gini impurity measures the impurity of a node in a decision tree.
            It is defined as 1 - sum(p_i^2) where p_i is the proportion of samples
            of class i in the node.

            ## Args:
                **labels**: *np.ndarray*
                Array of class labels.

            ## Returns:
                **float**: *The Gini impurity value.*

            ## Raises:
                **None**
            """
            if len(labels) == 0:
                return 0.0

            labels = labels.astype(np.int32)
            max_label = labels.max() if len(labels) > 0 else 0
            counts = np.bincount(labels, minlength=max_label + 1)
            probs = counts / len(labels)
            gini = 1.0 - np.sum(probs ** 2)
            return gini
        
        @staticmethod
        def log_loss_impurity(labels: np.ndarray) -> float:
            """
            Calculate the log loss (cross-entropy) for a set of labels.

            Log loss measures the impurity of a node in a decision tree.
            It is defined as -sum(p_i * log(p_i)) where p_i is the proportion of samples
            of class i in the node.

            ## Args:
                **labels**: *np.ndarray* 
                Array of class labels.

            ## Returns:
                **float**: *The log loss value.*

            ## Raises:
                **None**
            """
            if len(labels) == 0:
                return 0.0

            labels = labels.astype(np.int32)
            max_label = labels.max() if len(labels) > 0 else 0
            counts = np.bincount(labels, minlength=max_label + 1)
            probs = counts / len(labels)

            log_loss_val = 0.0
            for p in probs:
                if p > 0:
                    log_loss_val -= p * np.log(p)

            return log_loss_val
        
        @staticmethod
        def entropy_impurity(labels: np.ndarray) -> float:
            """
            Calculate the entropy for a set of labels.

            Entropy measures the impurity of a node in a decision tree.
            It is defined as -sum(p_i * log2(p_i)) where p_i is the proportion of samples
            of class i in the node.

            ## Args:
                **labels**: *np.ndarray*
                Array of class labels.

            ## Returns:
                **float**: *The entropy value.*

            ## Raises:
                **None**
            """
            if len(labels) == 0:
                return 0.0

            labels = labels.astype(np.int32)
            max_label = labels.max() if len(labels) > 0 else 0
            counts = np.bincount(labels, minlength=max_label + 1)
            probs = counts / len(labels)

            entropy_val = 0.0
            for p in probs:
                if p > 0:
                    entropy_val -= p * np.log2(p)
            return entropy_val