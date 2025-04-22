import logging
from zenml import step
import pandas as pd
from src.model_evaluation import MSE, R2Score, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import numpy as np
from zenml.client import Client
import mlflow


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame) -> Tuple[Annotated[float, "mse"], Annotated[float, "r2_score"], Annotated[float, "rmse"]]: 
    """Evaluate the model on test data."""
    try:
        logging.info("Starting model evaluation...")
        y_pred = model.predict(X_test)
        
        mse = MSE()
        mse = mse.evaluate(np.ravel(y_test), np.ravel(y_pred))
        mlflow.log_metric("mse", mse)

        r2_score = R2Score()
        r2_score = r2_score.evaluate(np.ravel(y_test), np.ravel(y_pred))
        mlflow.log_metric("r2_score", r2_score)

        rmse = RMSE()
        rmse = rmse.evaluate(np.ravel(y_test), np.ravel(y_pred))
        mlflow.log_metric("rmse", rmse)

        logging.info("Model evaluation completed successfully.")
        return mse, r2_score, rmse
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise e
    