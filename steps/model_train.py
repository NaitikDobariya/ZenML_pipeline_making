import logging

import pandas as pd
import mlflow
from zenml import step
from zenml.client import Client

from src.model_development import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

# experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker = experiment_tracker.name)
@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    
    try:
        model = None
        if config.model_name == "LinearRegression":
            # mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, Y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e