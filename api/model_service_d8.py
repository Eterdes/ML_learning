import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

class ModelService:
    # Маппинг строка -> класс модели
    MODEL_CLASSES = {
        "logreg": LogisticRegression,
        "random_forest": RandomForestClassifier
    }
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.pipeline = None
        self.metadata = {}
        
    def get_model_by_type(self, model_type: str, hyperparameters: dict):
        """Создаёт экземпляр модели по типу и параметрам"""
        if model_type not in self.MODEL_CLASSES:
            raise ValueError(f"Unknown model_type: {model_type}. Available: {list(self.MODEL_CLASSES.keys())}")
        
        model_class = self.MODEL_CLASSES[model_type]
        return model_class(**hyperparameters)
    
    def train_churn_model(self, X_train, y_train, model_type: str = "logreg", hyperparameters: dict = None):
        """Обучает модель с заданным типом и параметрами"""
        if hyperparameters is None:
            hyperparameters = {}
        
        # Определяем категориальные и числовые признаки
        categorical_features = ['region', 'device_type', 'payment_method']
        numeric_features = ['monthly_fee', 'usage_hours', 'support_requests', 
                           'account_age_months', 'failed_payments', 'autopay_enabled']
        
        # Создаём препроцессор
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Создаём модель
        model = self.get_model_by_type(model_type, hyperparameters)
        
        # Создаём pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Обучаем
        self.pipeline.fit(X_train, y_train)
        
        # Сохраняем конфигурацию в метаданных
        self.metadata['model_type'] = model_type
        self.metadata['hyperparameters'] = hyperparameters
        
    def evaluate_model(self, X_test, y_test):
        """Оценка модели"""
        if self.pipeline is None:
            return {"error": "Model not trained"}
        
        y_pred = self.pipeline.predict(X_test)
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred))
        }
        
        return metrics
    
    def save_churn_model(self):
        """Сохраняет модель"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        
    def load_churn_model(self):
        """Загружает модель"""
        if self.model_path.exists():
            self.pipeline = joblib.load(self.model_path)
            
    def save_metadata(self, metrics: dict):
        """Сохраняет метаданные модели"""
        metadata_path = self.model_path.parent / "model_metadata.json"
        
        # Объединяем метрики с существующими метаданными
        self.metadata.update({"metrics": metrics})
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def load_metadata(self):
        """Загружает метаданные"""
        metadata_path = self.model_path.parent / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                
    def get_status(self):
        """Возвращает статус модели"""
        return {
            "model_loaded": self.pipeline is not None,
            "model_type": self.metadata.get("model_type", "unknown"),
            "hyperparameters": self.metadata.get("hyperparameters", {}),
            "metrics": self.metadata.get("metrics", {})
        }
    
    def predict(self, features):
        """Предсказание для одного объекта"""
        df = pd.DataFrame([features.dict()])
        prediction = self.pipeline.predict(df)[0]
        probability = self.pipeline.predict_proba(df)[0][1]
        
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability)
        }
