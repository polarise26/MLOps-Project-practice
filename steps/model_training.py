import logging
import pandas as pd
import mlflow

from zenml import step
from sklearn.base import RegressorMixin
from src.model_dev import LinearRegressionModel, LightGBMModel, RandomForestModel, XGBoostModel, HyperparameterTuner
from .config import ModelNameConfig


@step
def trainModel(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:

    try:
        model = None
        if config.model_name = "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        elif config.model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            model = LightGBMModel()
        elif config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBoostModel()
        pass

        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        if config.fine_tuning == True:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            trained_model = model.train(x_train, y_train)
            return trained_model
    except Exception as e:
        logging.error(f"Error while training model: {e}")
        raise e
