import logging
import pandas as pd

from zenml import step

class IngestData:
    '''
    Class to ingest data from the data path
    '''
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def getData(self):
        logging.info(f"Ingesting data from {self.data_path}.")
        return pd.read_csv(self.data_path)
    
@step
def ingestData(data_path: str) -> pd.DataFrame:
    '''
    Ingests the data from the data_path.
    
    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data
    '''
    try:
        ingest_data = IngestData(data_path)
        
        return ingest_data.getData()
    except Exception as e:
        logging.error(f"Error while data ingestion: {e}")
        raise e