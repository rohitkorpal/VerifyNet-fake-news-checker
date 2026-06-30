class BaseModel:
    """
    Base class for all classifiers. Ensures standard scikit-learn style interface.
    """
    def fit(self, X, y):
        raise NotImplementedError("Each model must implement a fit method.")

    def predict(self, X):
        raise NotImplementedError("Each model must implement a predict method.")

    def score(self, X, y):
        """
        Returns the accuracy of the model on the given dataset.
        """
        predictions = self.predict(X)
        correct = sum(p == act for p, act in zip(predictions, y))
        return correct / len(y) if len(y) > 0 else 0.0
