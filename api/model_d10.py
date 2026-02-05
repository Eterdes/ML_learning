from pydantic import BaseModel
from typing import Dict, Optional, Union, List

class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int

class DatasetRowChurn(FeatureVectorChurn):
    churn: int

class PredictionResponseChurn(BaseModel):
    churn_prediction: int
    churn_probability: float

class TrainingConfigChurn(BaseModel):
    model_type: str = "logreg"
    hyperparameters: Dict = {}

class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict = {}