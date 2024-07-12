import logging
import pandas as pd

from zenml import step

@step
def cleanData(df: pd.DataFrame):
    pass