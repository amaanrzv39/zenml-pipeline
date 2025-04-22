import logging
from abc import ABC, abstractmethod

class Model(ABC):
    """Abstract base class for model training"""

    @abstractmethod
    def train(self, X_train, y_train) -> None:
        """Train the model."""
        pass


class CatboostRegressionModel(Model):
    """Catboost Regressor model"""

    def train(self, X_train, y_train, **params) -> None:
        """Train the Catboost Regressor model."""
        from catboost import CatBoostRegressor
        try:
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train)
            logging.info("Model training completed successfully.")
            return model
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e