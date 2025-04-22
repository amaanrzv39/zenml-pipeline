import logging
from zenml import step
import pandas as pd
from sklearn.base import RegressorMixin
from src.model_training import LinearRegressionModel, CatBoostModel, RandomForestModel, HyperparameterTuner
from .config import ModelNameConfig
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

config = ModelNameConfig()

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame) -> RegressorMixin:
    """Train the model on ingested data."""

    Model = None
    try:
        logging.info("Starting model training...")
        if config.model_name == "catboost":
            mlflow.catboost.autolog()
            model = CatBoostModel()
        elif config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            raise ValueError(f"Model {config.model_name} is not supported.")

        tuner = HyperparameterTuner(model, X_train, y_train, X_test, y_test)

        if config.fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(X_train, y_train, **best_params)
        else:
            trained_model = model.train(X_train, y_train)
        return trained_model
            
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise e
   