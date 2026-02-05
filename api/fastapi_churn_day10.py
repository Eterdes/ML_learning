from api.dataset_service import DatasetService
from api.model_service_d9 import ModelService
from api.model_d10 import FeatureVectorChurn, PredictionResponseChurn, TrainingConfigChurn, ErrorResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pathlib import Path
from typing import Union, List


app = FastAPI()

project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "data" / "churn_dataset.csv"
model_path = project_root / "models" / "churn_model.joblib"

dataset_service = DatasetService(csv_path)
model_service = ModelService(model_path)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Обработка HTTPException"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "details": {}
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Обработка ошибок валидации Pydantic"""
    errors = exc.errors()
    return JSONResponse(
        status_code=422,
        content={
            "code": "VALIDATION_ERROR",
            "message": "Invalid input data",
            "details": {
                "errors": errors
            }
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Обработка ValueError"""
    return JSONResponse(
        status_code=400,
        content={
            "code": "VALUE_ERROR",
            "message": str(exc),
            "details": {}
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Обработка всех остальных ошибок"""
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "Internal server error",
            "details": {
                "error_type": type(exc).__name__,
                "error_message": str(exc)
            }
        }
    )


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


@app.post("/model/train",
    responses={
        200: {"description": "Model trained successfully"},
        400: {"model": ErrorResponse, "description": "Training error"},
    }
)
def train_model(config: TrainingConfigChurn = None):

    if dataset_service.df is None:
        raise HTTPException(
            status_code=400, 
            detail="Dataset not loaded"
        )
    
    if dataset_service.df.empty:
        raise HTTPException(
            status_code=400, 
            detail="Dataset is empty"
        )
    
    if dataset_service.X_train is None:
        raise HTTPException(
            status_code=400, 
            detail="Dataset not split. Call /dataset/split-info first"
        )
    
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
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Training failed: {str(e)}"
        )
    

@app.get("/model/status")
def model_status():
    return model_service.get_status()


@app.post("/predict", 
    response_model=Union[PredictionResponseChurn, List[PredictionResponseChurn]],
    responses={
        200: {"description": "Prediction successful"},
        400: {"model": ErrorResponse, "description": "Model not trained or invalid data"},
        422: {"model": ErrorResponse, "description": "Invalid input format"}
    }
)
async def predict(features: Union[FeatureVectorChurn, List[FeatureVectorChurn]]):
    if model_service.pipeline is None:
        raise HTTPException(
            status_code=400, 
            detail="Model not trained. Train model first using /model/train"
        )
    
    try:
        if isinstance(features, list):
            results = [model_service.predict(f) for f in features]
            return results
        else:
            result = model_service.predict(features)
            return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model/schema")
def model_schema():
    """Возвращает схему признаков модели"""
    return model_service.get_feature_schema()