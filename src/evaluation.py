import logging
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):

    @abstractmethod
    def calculateScores(self, y_true, y_pred):
        pass


class MSE(Evaluation):

    def calculateScores(self, y_true, y_pred):
        logging.info("Calculating MSE...")
        mse = mean_squared_error(y_true, y_pred)
        logging.info(f"MSE: {mse}")
        return mse


class R2(Evaluation):

    def calculateScores(self, y_true, y_pred):
        logging.info("Calculating R2 score...")
        r2 = r2_score(y_true, y_pred)
        logging.info(f"R2 Score: {r2}")
        return r2


class RMSE(Evaluation):

    def calculateScores(self, y_true, y_pred):
        logging.info("Calculating RMSE...")
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        logging.info(f"RMSE: {rmse}")
        return rmse
