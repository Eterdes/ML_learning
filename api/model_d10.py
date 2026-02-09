from pydantic import BaseModel, Field, ConfigDict, conint, confloat, constr
from typing import Any, Dict


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monthly_fee: confloat(ge=0)
    usage_hours: confloat(ge=0)
    support_requests: conint(ge=0)
    account_age_months: conint(ge=0)
    failed_payments: conint(ge=0)

    region: constr(min_length=1)
    device_type: constr(min_length=1)
    payment_method: constr(min_length=1)

    autopay_enabled: conint(ge=0, le=1)


class DatasetRowChurn(FeatureVectorChurn):
    churn: conint(ge=0, le=1)


class PredictionResponseChurn(BaseModel):
    churn_prediction: int
    churn_probability: float


from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict

class TrainingConfigChurn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {   # ⬅️ ВАЖНО: example, НЕ examples
                "model_type": "logreg",
                "hyperparameters": {}
            }
        }
    )

    model_type: str = "logreg"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)