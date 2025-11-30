class ModelLike:
    """
    Helper for checking model compability
    """
    def __init__(self, model) -> None:
        """
        Check model compability for code robustness.

        ## Args:
          **model**: *Any*
          model's class.

        ## Returns:
          **None**

        ## Raises:
          **None**
        """
        self.model = model

    def is_standard(self) -> bool:
        """
        Check if the model has standard classic linear model attribute.

        ## Args:
          **None**

        ## Returns:
          **bool**: *Model's 'standard' status.*

        ## Raises:
          **None**
        """
        return True if all([
            hasattr(self.model, 'loss_history'), 
            hasattr(self.model, 'weights'), 
            hasattr(self.model, 'b'),
            hasattr(self.model, 'fit'),
            hasattr(self.model, 'predict')]) else False

    def is_regressor(self) -> bool:
        """
        Check if the model is regressor.

        ## Args:
          **None**

        ## Returns:
          **bool**: *Model 'regressor' status.*

        ## Raise:
          **None**
        """
        return True if self.is_standard() else False
    
    def is_classifier(self) -> bool:
        """
        Check if the model is classifier.

        ## Args:
          **None**

        ## Returns:
          **bool**: *Model 'classifier' status.*

        ## Raise:
          **None**
        """
        return True if all([
            self.is_standard(), 
            hasattr(self.model, 'predict_proba')]) else False
