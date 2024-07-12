import logging
import pandas as pd

from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated


@step
def cleanData(df: pd.DataFrame) -> tuple(
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
):
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaner = DataCleaning(df, process_strategy)
        processed_data = data_cleaner.handleData()

        dividing_strategy = DataDivideStrategy()
        data_cleaner = DataCleaning(processed_data, dividing_strategy)
        X_train, X_test, y_train, y_test = data_cleaner.handleData()
        logging.info("Data cleaning performed.")

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error while cleaning data: {e}")
        raise e
