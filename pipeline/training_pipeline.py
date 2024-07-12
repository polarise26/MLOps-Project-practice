from zenml import pipeline

from steps.ingest_data import ingestData
from steps.clean_data import cleanData
from steps.model_training import trainModel
from steps.evaluation import evaluateModel

@pipeline
def trainingPipeline(data_path: str):
    df = ingestData(data_path)
    cleanData(df)
    trainModel(df)
    evaluateModel(df)
