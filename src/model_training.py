import logging
from abc import ABC, abstractmethod
import optuna
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

class Model(ABC):
    """Abstract base class for model training"""

    @abstractmethod
    def train(self, X_train, y_train) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            x_train: Training data
            y_train: Target data
            x_test: Testing data
            y_test: Testing target
        """
        pass


class LinearRegressionModel(Model):
    """Linear Regression model"""

    def train(self, X_train, y_train, **params):
        try:
            model = LinearRegression(**params)
            model.fit(X_train, y_train)
            logging.info("Model training completed successfully.")
            return model
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e
        
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        model = self.train(x_train, y_train)
        return model.score(x_test, y_test)

class RandomForestModel(Model):
    """Linear Regression model"""

    def train(self, X_train, y_train, **params):
        try:
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            logging.info("Model training completed successfully.")
            return model
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e
        
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        model = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        return model.score(x_test, y_test)
        
class CatBoostModel(Model):
    """Linear Regression model"""

    def train(self, X_train, y_train, **params):
        try:
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train)
            logging.info("Model training completed successfully.")
            return model
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e
        
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        model = self.train(x_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        return model.score(x_test, y_test)
    
class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=n_trials)
        return study.best_trial.params