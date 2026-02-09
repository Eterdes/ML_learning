from pathlib import Path
from typing import Union, List, Optional
from fastapi import FastAPI, HTTPException, Request, Query, Body
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from api.dataset_service import DatasetService
from api.model_service_d10 import ModelService, ModelNotTrainedError
from api.model_d10 import FeatureVectorChurn, PredictionResponseChurn, TrainingConfigChurn, ErrorResponse
from api.training_history_d11 import TrainingHistoryService

import logging

logger = logging.getLogger("churn_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI()

project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "data" / "churn_dataset.csv"
model_path = project_root / "models" / "churn_model.joblib"
history_path = project_root / "models" / "training_history.json"
training_history = TrainingHistoryService(history_path)

dataset_service = DatasetService(csv_path)
model_service = ModelService(model_path)
                             
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": f"HTTP_{exc.status_code}", "message": exc.detail, "details": {}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "code": "VALIDATION_ERROR",
            "message": "Invalid input data",
            "details": {"errors": exc.errors()},
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"code": "VALUE_ERROR", "message": str(exc), "details": {}},
    )


@app.exception_handler(ModelNotTrainedError)
async def model_not_trained_handler(request: Request, exc: ModelNotTrainedError):
    return JSONResponse(
        status_code=400,
        content={"code": "MODEL_NOT_TRAINED", "message": str(exc), "details": {}},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_ERROR", "message": "Internal server error", "details": {}},
    )


@app.on_event("startup")
def on_startup():
    logger.info("startup: loading dataset...")
    try:
        dataset_service.load()
        logger.info("startup: dataset loaded")
    except Exception:
        logger.exception("startup: dataset load failed")

    logger.info("startup: splitting dataset...")
    try:
        dataset_service.split_data()
        logger.info("startup: dataset split done")
    except Exception:
        logger.exception("startup: dataset split failed")

    logger.info("startup: loading model + metadata...")
    try:
        model_service.load_churn_model()
        model_service.load_metadata()
        logger.info("startup: model/metadata loaded")
    except Exception:
        logger.exception("startup: model/metadata load failed")


@app.get(
    "/dataset/split-info",
    responses={
        200: {
            "description": "Train/test split info",
            "content": {
                "application/json": {
                    "example": {
                        "train_size": 1600,
                        "test_size": 400,
                        "train_churn_distribution": {"0": 1278, "1": 322},
                        "test_churn_distribution": {"0": 319, "1": 81},
                    }
                }
            },
        },
        400: {
            "description": "Dataset not split yet",
            "content": {
                "application/json": {
                    "example": {
                        "code": "HTTP_400",
                        "message": "Dataset not split. Run split_data first",
                        "details": {}
                    }
                }
            },
        },
        500: {
            "description": "Internal error",
            "content": {
                "application/json": {
                    "example": {
                        "code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                        "details": {}
                    }
                }
            },
        },
    },
)


def split_info():
    # защита от AttributeError/None (чтобы не было 500)
    if not hasattr(dataset_service, "X_train") or dataset_service.X_train is None:
        raise HTTPException(status_code=400, detail="Dataset not split. Run split_data first")

    return dataset_service.get_split_info()



@app.get(
    "/dataset/info",
    responses={
        200: {
            "description": "Dataset basic info",
            "content": {
                "application/json": {
                    "example": {
                        "rows": 2000,
                        "cols": 10,
                        "columns": [
                            "monthly_fee",
                            "usage_hours",
                            "support_requests",
                            "account_age_months",
                            "failed_payments",
                            "region",
                            "device_type",
                            "payment_method",
                            "autopay_enabled",
                            "churn"
                        ],
                        "churn_distribution": {"0": 1597, "1": 403}
                    }
                }
            },
        },
        400: {
            "description": "Dataset not loaded",
            "content": {
                "application/json": {
                    "example": {
                        "code": "HTTP_400",
                        "message": "Dataset not loaded",
                        "details": {}
                    }
                }
            },
        },
        500: {
            "description": "Internal error",
            "content": {
                "application/json": {
                    "example": {
                        "code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                        "details": {}
                    }
                }
            },
        },
    },
)
def info():
    if dataset_service.df is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    return dataset_service.info()


@app.get(
    "/dataset/preview",
    responses={
        200: {
            "description": "Dataset preview (first N rows)",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "monthly_fee": 9.99,
                            "usage_hours": 27.92,
                            "support_requests": 1,
                            "account_age_months": 14,
                            "failed_payments": 1,
                            "region": "america",
                            "device_type": "desktop",
                            "payment_method": "card",
                            "autopay_enabled": 1,
                            "churn": 1
                        },
                        {
                            "monthly_fee": 19.99,
                            "usage_hours": 21.48,
                            "support_requests": 2,
                            "account_age_months": 1,
                            "failed_payments": 0,
                            "region": "europe",
                            "device_type": "mobile",
                            "payment_method": "paypal",
                            "autopay_enabled": 0,
                            "churn": 0
                        }
                    ]
                }
            },
        },
        400: {
            "description": "Dataset not loaded or invalid n",
            "content": {
                "application/json": {
                    "example": {
                        "code": "HTTP_400",
                        "message": "Dataset not loaded",
                        "details": {}
                    }
                }
            },
        },
        422: {
            "description": "Validation error (invalid query param)",
            "content": {
                "application/json": {
                    "example": {
                        "code": "VALIDATION_ERROR",
                        "message": "Invalid input data",
                        "details": {
                            "errors": [
                                {
                                    "loc": ["query", "n"],
                                    "msg": "Input should be greater than 0",
                                    "type": "greater_than"
                                }
                            ]
                        }
                    }
                }
            },
        },
        500: {
            "description": "Internal error",
            "content": {
                "application/json": {
                    "example": {
                        "code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                        "details": {}
                    }
                }
            },
        },
    },
)
def preview(n: int = Query(default=10, ge=1, le=100)):
    if dataset_service.df is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    return dataset_service.preview(n)


# ---------- model endpoints ----------

@app.post(
    "/model/train",
    responses={
        200: {
            "description": "Model trained successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "trained",
                        "model_type": "logreg",
                        "hyperparameters": {},
                        "metrics": {
                            "accuracy": 0.82,
                            "precision": 0.74,
                            "recall": 0.68,
                            "f1_score": 0.71
                        }
                    }
                }
            },
        },
        400: {
            "description": "Training error (dataset or configuration problem)",
            "content": {
                "application/json": {
                    "example": {
                        "code": "VALUE_ERROR",
                        "message": "Dataset is empty",
                        "details": {}
                    }
                }
            },
        },
        422: {
            "description": "Invalid input format",
            "content": {
                "application/json": {
                    "example": {
                        "code": "VALIDATION_ERROR",
                        "message": "Invalid input data",
                        "details": {
                            "errors": [
                                {
                                    "loc": ["body", "model_type"],
                                    "msg": "Input should be a valid string",
                                    "type": "string_type"
                                }
                            ]
                        }
                    }
                }
            },
        },
        500: {
            "description": "Internal error",
            "content": {
                "application/json": {
                    "example": {
                        "code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                        "details": {}
                    }
                }
            },
        },
    },
)
def train_model(
    config: TrainingConfigChurn = Body(default_factory=TrainingConfigChurn)
):
    logger.info(
        "train: request | model_type=%s | hyperparameters=%s",
        config.model_type,
        config.hyperparameters,
    )

    if dataset_service.df is None:
        logger.warning("train: failed | reason=dataset_not_loaded")
        raise HTTPException(status_code=400, detail="Dataset not loaded")

    if dataset_service.df.empty:
        logger.warning("train: failed | reason=dataset_empty")
        raise HTTPException(status_code=400, detail="Dataset is empty")

    if not hasattr(dataset_service, "X_train") or dataset_service.X_train is None:
        logger.warning("train: failed | reason=dataset_not_split")
        raise HTTPException(
            status_code=400,
            detail="Dataset not split. Run split_data first"
        )

    try:
        logger.info("train: start")

        # обучение
        model_service.train_churn_model(
            dataset_service.X_train,
            dataset_service.y_train,
            model_type=config.model_type,
            hyperparameters=config.hyperparameters,
        )

        # метрики + сохранение
        metrics = model_service.evaluate_model(
            dataset_service.X_test,
            dataset_service.y_test
        )
        model_service.save_churn_model()
        model_service.save_metadata(metrics)

        training_history.add_record(
            model_type=config.model_type,
            hyperparameters=config.hyperparameters,
            metrics=metrics
        )

        logger.info("train: done | metrics=%s", metrics)

        return {
            "status": "trained",
            "model_type": config.model_type,
            "hyperparameters": config.hyperparameters,
            "metrics": metrics,
        }

    except Exception:
        logger.exception("train: error")
        raise


@app.get(
    "/model/status",
    responses={
        200: {
            "description": "Model status info",
            "content": {
                "application/json": {
                    "example": {
                        "model_loaded": True,
                        "model_type": "logreg",
                        "hyperparameters": {},
                        "metrics": {
                            "accuracy": 0.82,
                            "precision": 0.74,
                            "recall": 0.68,
                            "f1_score": 0.71
                        }
                    }
                }
            },
        },
        500: {
            "description": "Internal error",
            "content": {
                "application/json": {
                    "example": {
                        "code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                        "details": {}
                    }
                }
            },
        },
    },
)
def model_status():
    return model_service.get_status()


@app.post(
    "/predict",
    response_model=Union[PredictionResponseChurn, List[PredictionResponseChurn]],
    responses={
        200: {
            "description": "Prediction successful",
            "content": {
                "application/json": {
                    "example": {
                        "churn_prediction": 0,
                        "churn_probability": 0.23
                    }
                }
            },
        },
        400: {
            "description": "Model not trained or invalid data",
            "content": {
                "application/json": {
                    "example": {
                        "code": "MODEL_NOT_TRAINED",
                        "message": "Model not trained. Train model first using /model/train",
                        "details": {}
                    }
                }
            },
        },
        422: {
            "description": "Invalid input format",
            "content": {
                "application/json": {
                    "example": {
                        "code": "VALIDATION_ERROR",
                        "message": "Invalid input data",
                        "details": {
                            "errors": [
                                {
                                    "loc": ["body", "monthly_fee"],
                                    "msg": "Input should be a valid number",
                                    "type": "float_parsing"
                                }
                            ]
                        }
                    }
                }
            },
        },
        500: {
            "description": "Internal error",
            "content": {
                "application/json": {
                    "example": {
                        "code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                        "details": {}
                    }
                }
            },
        },
    },
)
async def predict(features: Union[FeatureVectorChurn, List[FeatureVectorChurn]]):

    # ---- лог входа ----
    if isinstance(features, list):
        logger.info("predict: batch request | size=%d", len(features))
    else:
        logger.info(
            "predict: request | monthly_fee=%s usage_hours=%s support_requests=%s failed_payments=%s region=%s device=%s autopay=%s",
            features.monthly_fee,
            features.usage_hours,
            features.support_requests,
            features.failed_payments,
            features.region,
            features.device_type,
            features.autopay_enabled,
        )

    try:
        # ---- существующая логика ----
        if isinstance(features, list):
            results = [model_service.predict(f) for f in features]

            logger.info(
                "predict: batch response | size=%d | positives=%d",
                len(results),
                sum(r.get("churn_prediction", 0) for r in results),
            )
            return results

        result = model_service.predict(features)

        logger.info(
            "predict: response | pred=%s | proba=%.6f",
            result.get("churn_prediction"),
            float(result.get("churn_probability", 0.0)),
        )
        return result

    except Exception:
        logger.exception("predict: error")
        raise


@app.get(
    "/model/schema",
    responses={
        200: {
            "description": "Feature schema used by the model",
            "content": {
                "application/json": {
                    "example": {
                        "numeric_features": [
                            "monthly_fee",
                            "usage_hours",
                            "support_requests",
                            "account_age_months",
                            "failed_payments",
                            "autopay_enabled"
                        ],
                        "categorical_features": [
                            "region",
                            "device_type",
                            "payment_method"
                        ],
                        "feature_types": {
                            "monthly_fee": "float",
                            "usage_hours": "float",
                            "support_requests": "int",
                            "account_age_months": "int",
                            "failed_payments": "int",
                            "autopay_enabled": "int",
                            "region": "str",
                            "device_type": "str",
                            "payment_method": "str"
                        },
                        "total_features": 9
                    }
                }
            },
        },
        500: {
            "description": "Internal error",
            "content": {
                "application/json": {
                    "example": {
                        "code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                        "details": {}
                    }
                }
            },
        },
    },
)
def model_schema():
    # если вдруг IDE/код не туда смотрит — будет понятная 500, а не AttributeError
    if not hasattr(model_service, "get_feature_schema"):
        raise HTTPException(status_code=500, detail="ModelService.get_feature_schema is missing")
    return model_service.get_feature_schema()


@app.get(
    "/model/metrics",
    responses={
        200: {
            "description": "Training metrics history for churn model",
            "content": {
                "application/json": {
                    "example": {
                        "last": {
                            "accuracy": 0.80,
                            "precision": 1.0,
                            "recall": 0.0123,
                            "f1": 0.0244,
                            "roc_auc": 0.86
                        },
                        "history": []
                    }
                }
            },
        },
        422: {
            "description": "Validation error (invalid query param)",
            "content": {
                "application/json": {
                    "example": {
                        "code": "VALIDATION_ERROR",
                        "message": "Invalid input data",
                        "details": {
                            "errors": [
                                {
                                    "loc": ["query", "limit"],
                                    "msg": "Input should be greater than or equal to 1",
                                    "type": "greater_than_equal"
                                }
                            ]
                        }
                    }
                }
            },
        },
        500: {
            "description": "Internal error",
            "content": {
                "application/json": {
                    "example": {
                        "code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                        "details": {}
                    }
                }
            },
        },
    },
)
def model_metrics(
    model_type: Optional[str] = Query(default=None),
    limit: int = Query(default=5, ge=1, le=50),
):
    history = training_history.get_history(model_type=model_type, limit=limit)
    last_metrics = history[-1]["metrics"] if history else {}
    return {"last": last_metrics, "history": history}


@app.get(
    "/health",
    responses={
        200: {
            "description": "Service health status",
            "content": {
                "application/json": {
                    "examples": {
                        "healthy": {
                            "summary": "Service fully operational",
                            "value": {
                                "status": "ok",
                                "dataset_loaded": True,
                                "split_ready": True,
                                "model_loaded": True,
                            },
                        },
                        "degraded_no_model": {
                            "summary": "Service running, model not trained",
                            "value": {
                                "status": "degraded",
                                "dataset_loaded": True,
                                "split_ready": True,
                                "model_loaded": False,
                            },
                        },
                        "degraded_no_dataset": {
                            "summary": "Service running, dataset not loaded",
                            "value": {
                                "status": "degraded",
                                "dataset_loaded": False,
                                "split_ready": False,
                                "model_loaded": False,
                            },
                        },
                    }
                }
            },
        }
    },
)
@app.get("/health")
def health():
    dataset_loaded = getattr(dataset_service, "df", None) is not None
    split_ready = getattr(dataset_service, "X_train", None) is not None

    status = model_service.get_status()  # уже знает правду
    model_loaded = bool(status.get("model_loaded", False))

    overall_ok = dataset_loaded and model_loaded

    return {
        "status": "ok" if overall_ok else "degraded",
        "dataset_loaded": dataset_loaded,
        "split_ready": split_ready,
        "model_loaded": model_loaded
    }