import pandas as pd
import numpy as np
import pytest

from api.model_service_d10 import ModelService, ModelNotTrainedError
from api.model_d10 import FeatureVectorChurn

def test_model_train_evaluate_predict(tmp_path):
    rng = np.random.default_rng(0)

    X = pd.DataFrame({
        "monthly_fee": rng.uniform(5, 50, 120),
        "usage_hours": rng.uniform(0, 100, 120),
        "support_requests": rng.integers(0, 5, 120),
        "account_age_months": rng.integers(0, 36, 120),
        "failed_payments": rng.integers(0, 4, 120),
        "region": rng.choice(["europe", "america"], 120),
        "device_type": rng.choice(["mobile", "desktop"], 120),
        "payment_method": rng.choice(["card", "paypal"], 120),
        "autopay_enabled": rng.integers(0, 2, 120),
    })
    y = ((X["monthly_fee"] > 30).astype(int) + (X["failed_payments"] > 1).astype(int) > 1).astype(int)

    X_train, X_test = X.iloc[:100], X.iloc[100:]
    y_train, y_test = y.iloc[:100], y.iloc[100:]

    svc = ModelService(tmp_path / "churn_model.joblib")

    svc.train_churn_model(X_train, y_train, model_type="logreg", hyperparameters={"max_iter": 200})
    metrics = svc.evaluate_model(X_test, y_test)

    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(metrics.keys())

    fv = FeatureVectorChurn(
        monthly_fee=19.99,
        usage_hours=10.0,
        support_requests=1,
        account_age_months=5,
        failed_payments=0,
        region="europe",
        device_type="mobile",
        payment_method="card",
        autopay_enabled=1,
    )
    pred = svc.predict(fv)
    assert pred["churn_prediction"] in (0, 1)
    assert 0.0 <= pred["churn_probability"] <= 1.0

def test_predict_without_training_raises(tmp_path):
    svc = ModelService(tmp_path / "churn_model.joblib")
    with pytest.raises(ModelNotTrainedError):
        # features не важны, упадет раньше
        class Dummy:
            def model_dump(self): return {}
        svc.predict(Dummy())