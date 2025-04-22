import logging
from zenml import step
import pandas as pd
from sklearn.base import RegressorMixin
from src.model_training import CatboostRegressionModel
from .config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig) -> RegressorMixin:
    """Train the model on ingested data."""

