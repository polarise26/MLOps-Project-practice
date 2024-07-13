import logging
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from typing import Union


class DataStrategy(ABC):

    @abstractmethod
    def handleData(self, df: pd.DataFrame):
        pass


class DataPreProcessStrategy(DataStrategy):

    def handleData(self, df: pd.DataFrame):
        try:
            df = df.drop(
                ["order_purchase_timestamp",
                 "order_approved_at",
                 "order_delivered_carrier_date",
                 "order_delivered_customer_date",
                 "order_estimated_delivery_date",
                 "customer_zip_code_prefix",
                 "order_item_id"
                 ],
                axis=1)
            df["product_weight_g"].fillna(
                df["product_weight_g"].median(), inplace=True)
            df["product_length_cm"].fillna(
                df["product_length_cm"].median(), inplace=True)
            df["product_height_cm"].fillna(
                df["product_height_cm"].median(), inplace=True)
            df["product_width_cm"].fillna(
                df["product_width_cm"].median(), inplace=True)
            # write "No review" in review_comment_message column
            df["review_comment_message"].fillna("No review", inplace=True)

            # for simplicity, we use only numerical data
            df = df.select_dtypes(include=[np.number])

            return df
        except Exception as e:
            logging.error(f"Error while preprocessing data: {e}")
            raise e


class DataDivideStrategy(DataStrategy):

    def handleData(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = df.drop(["review_score"], axis=1)
            y = df["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=26)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error dividing preprocessing data: {e}")
            raise e


class DataCleaning:
    def __init__(self, df: pd.DataFrame, strategy: DataStrategy):
        self.df = df
        self.strategy = strategy

    def handleData(self):
        try:
            return self.strategy.handleData(self.df)
        except Exception as e:
            logging.error(f"Error while cleaning data: {e}")
            raise e


if __name__ == "__main__":
    data = pd.read_csv(
        r"C:\Users\kannu\OneDrive\Documents\github-repos\MLOps-Project-practice\data\olist_customers_dataset.csv")
    data_cleaner = DataCleaning(data, DataPreProcessStrategy())
    data_cleaner.handleData()
