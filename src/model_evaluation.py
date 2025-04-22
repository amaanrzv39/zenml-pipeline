import logging
from abc import ABC, abstractmethod
import numpy as np


class ModelEvaluation(ABC):
    """Abstract base class for model evaluation."""

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Evaluate the model."""
        pass

class MSE(ModelEvaluation):
    """Mean Squared Error evaluation."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        try:
            mse = np.mean((y_true - y_pred) ** 2)
            logging.info(f"Mean Squared Error: {mse}")
            return np.round(mse,2)
        except Exception as e:
            logging.error(f"Error in MSE calculation: {e}")
            raise e

class R2Score(ModelEvaluation):
    """R2 Score evaluation."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R2 Score."""
        try:
            ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
            ss_residual = np.sum((y_true - y_pred) ** 2)
            r2_score = 1 - (ss_residual / ss_total)
            logging.info(f"R2 Score: {r2_score}")
            return np.round(r2_score,2)
        except Exception as e:
            logging.error(f"Error in R2 Score calculation: {e}")
            raise e 

class RMSE(ModelEvaluation):
    """Root Mean Squared Error evaluation."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        try:
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            logging.info(f"Root Mean Squared Error: {rmse}")
            return np.round(rmse,2)
        except Exception as e:
            logging.error(f"Error in RMSE calculation: {e}")
            raise e