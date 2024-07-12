from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    model_name: str = "LinearRegression"
    fine_tuning: bool = True
