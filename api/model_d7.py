from pydantic import BaseModel, Field
class FeatureVectorChurn(BaseModel):
    monthly_fee: float = Field(..., example=49.99)
    usage_hours: float = Field(..., example=120.5)
    support_requests: int = Field(..., example=2)
    account_age_months: int = Field(..., example=24)
    failed_payments: int = Field(..., example=0)
    region: str = Field(..., example="North")
    device_type: str = Field(..., example="Mobile")
    payment_method: str = Field(..., example="CreditCard")
    autopay_enabled: int = Field(..., example=1)

class DatasetRowChurn(FeatureVectorChurn):
    churn: int

class PredictionResponseChurn(BaseModel):
    churn_prediction: int = Field(..., example=0)
    churn_probability: float = Field(..., example=0.15)
    probabilities: list[float] = Field(..., example=[0.85, 0.15])
