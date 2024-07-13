import logging
import pandas as pd

from zenml import step
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from src.evaluation import MSE, R2, RMSE


@step
def evaluateModel(model: RegressorMixin, X_test, y_test) -> Tuple[Annotated[float, 'mse'], Annotated[float, 'r2_score'], Annotated[float, 'rmse']]:

    predictions = model.predict(X_test)

    mse_obj = MSE()
    mse = mse_obj.calculateScores(y_test, predictions)

    r2_obj = R2()
    r2 = r2_obj.calculateScores(y_test, predictions)

    rmse_obj = RMSE()
    rmse = rmse_obj.calculateScores(y_test, predictions)

    return mse, r2, rmse
