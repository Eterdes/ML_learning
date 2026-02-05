from api.dataset_service import DatasetService
from api.model_service_d8 import ModelService
from api.model_d8 import FeatureVectorChurn, PredictionResponseChurn, TrainingConfigChurn
from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import Union, List

app = FastAPI()

project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "data" / "churn_dataset.csv"
model_path = project_root / "models" / "churn_model.joblib"

dataset_service = DatasetService(csv_path)
model_service = ModelService(model_path)

@app.on_event("startup")
def on_startup():
    dataset_service.load()
    dataset_service.split_data()
    model_service.load_churn_model()
    model_service.load_metadata()

@app.get("/dataset/split-info")
def split_info():
    return dataset_service.get_split_info()

@app.get("/dataset/info")
def info():
    return dataset_service.info()

@app.get("/dataset/preview")
def preview(n: int = 10):
    return dataset_service.preview(n)

@app.post("/model/train")
def train_model(config: TrainingConfigChurn = None):
    if dataset_service.df is None:
        return {"error": "Dataset not loaded"}
    
    if dataset_service.df.empty:
        return {"error": "Dataset is empty"}
    
    if dataset_service.X_train is None:
        return {"error": "Dataset not split"}
    
    # Дефолтная конфигурация, если не передана
    if config is None:
        config = TrainingConfigChurn()
    
    try:
        model_service.train_churn_model(
            dataset_service.X_train,
            dataset_service.y_train,
            model_type=config.model_type,
            hyperparameters=config.hyperparameters
        )
        
        metrics = model_service.evaluate_model(
            dataset_service.X_test,
            dataset_service.y_test
        )

        model_service.save_churn_model()
        model_service.save_metadata(metrics)
        
        return {
            "status": "trained",
            "model_type": config.model_type,
            "hyperparameters": config.hyperparameters,
            "metrics": metrics
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/status")
def model_status():
    return model_service.get_status()

@app.post("/predict", response_model=Union[PredictionResponseChurn, List[PredictionResponseChurn]])
async def predict(features: Union[FeatureVectorChurn, List[FeatureVectorChurn]]):
    if model_service.pipeline is None:
        raise HTTPException(status_code=400, detail="Model not trained")
    
    if isinstance(features, list):
        results = [model_service.predict(f) for f in features]
        return results
    else:
        result = model_service.predict(features)
        return result