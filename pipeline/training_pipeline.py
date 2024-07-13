from zenml import pipeline

from steps.ingest_data import ingestData
from steps.clean_data import cleanData
from steps.model_training import trainModel
from steps.evaluation import evaluateModel

@pipeline
def trainingPipeline(data_path: str):
    df = ingestData(data_path)
    X_train, X_test, y_train, y_test = cleanData(df)
    model = trainModel(X_train, X_test, y_train, y_test)
    mse, r2_score, rmse = evaluateModel(model, X_test, y_test)
