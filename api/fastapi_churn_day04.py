from api.dataset_service import DatasetService
from fastapi import FastAPI
from pathlib import Path

app = FastAPI()

project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "data" / "churn_dataset.csv"

dataset_service = DatasetService(csv_path)

@app.on_event("startup")
def on_startup():
    dataset_service.load()
    dataset_service.split_data()

@app.get("/dataset/split-info")
def split_info():
    return dataset_service.get_split_info()

@app.get("/dataset/info")
def info():
    return dataset_service.info()

@app.get("/dataset/preview")
def preview(n: int = 10):
    return dataset_service.preview(n)

