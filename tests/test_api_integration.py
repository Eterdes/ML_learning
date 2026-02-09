import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.dataset_service import DatasetService 
from api.model_service_d10 import ModelService  
from api.training_history_d11 import TrainingHistoryService 

import api.fastapi_churn_day11 as main

@pytest.fixture
def client(tmp_path, synth_csv, monkeypatch):
    model_path = tmp_path / "models" / "churn_model.joblib"
    history_path = tmp_path / "models" / "training_history.json"

    # заменяем сервисы на тестовые
    ds = DatasetService(synth_csv)
    ms = ModelService(model_path)
    th = TrainingHistoryService(history_path)

    monkeypatch.setattr(main, "dataset_service", ds)
    monkeypatch.setattr(main, "model_service", ms)
    monkeypatch.setattr(main, "training_history", th)

    # важно: воспроизводимый split до обучения (у тебя в startup split_data без random_state)
    np.random.seed(42)

    with TestClient(main.app) as c:
        yield c

def test_full_pipeline_train_status_predict(client):
    # 1) predict до обучения -> ошибка MODEL_NOT_TRAINED (400)
    payload = {
        "monthly_fee": 9.99,
        "usage_hours": 12.3,
        "support_requests": 1,
        "account_age_months": 10,
        "failed_payments": 0,
        "region": "europe",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 1,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "MODEL_NOT_TRAINED"

    # 2) train
    r = client.post("/model/train", json={"model_type": "logreg", "hyperparameters": {"max_iter": 200}})
    assert r.status_code == 200
    train_body = r.json()
    assert train_body["status"] == "trained"
    assert "metrics" in train_body

    # 3) status
    r = client.get("/model/status")
    assert r.status_code == 200
    st = r.json()
    assert st["model_loaded"] is True
    assert st["model_type"] == "logreg"

    # 4) predict после обучения
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    pred = r.json()
    assert pred["churn_prediction"] in (0, 1)
    assert 0.0 <= pred["churn_probability"] <= 1.0

def test_validation_error_format(client):
    # monthly_fee должен быть числом >=0 (pydantic) :contentReference[oaicite:12]{index=12}
    r = client.post("/predict", json={"monthly_fee": "nope"})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert "errors" in body["details"]