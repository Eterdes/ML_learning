from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

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

@app.post("/predict")
async def root(features: FeatureVectorChurn):
    return features