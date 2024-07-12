import logging
import pandas as pd

from zenml import step

@step
def evaluateModel(df: pd.DataFrame) -> None:
    
    pass