import logging
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
import mlflow
from zenml import step
from zenml.client import Client

from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin

# experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker = experiment_tracker.name)
@step
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,

) -> Tuple[
    Annotated[float, "mse"],
    Annotated[float, "r2"],
    Annotated[float, "rmse"],
]:
    
    try:
        prediction = model.predict(X_test)

        mse_class = MSE()
        mse =  mse_class.calculate_scores(Y_test, prediction)
        # mlflow.log_metric("mse", mse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(Y_test, prediction)
        # mlflow.log_metric("r2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(Y_test, prediction)
        # mlflow.log_metric("rmse", rmse)

        return mse, r2, rmse
    
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e