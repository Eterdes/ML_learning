from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

class ModelService:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.pipeline = None
        self.metadata = None
        self.metadata_path = self.model_path.parent / "churn_model_metadata.json"


    def train_churn_model(self, X_train, y_train):
    
        numeric_features = [
            "monthly_fee", "usage_hours", "support_requests",
            "account_age_months", "failed_payments", "autopay_enabled"
        ]
        
        categorical_features = ["region", "device_type", "payment_method"]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        
        self.pipeline = pipeline
    
    def save_churn_model(self):
        if self.pipeline is None:
            raise RuntimeError("No model to save!")
        
        model_dir = self.model_path.parent 
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, self.model_path)

    def load_churn_model(self):
        if not self.model_path.exists():
            return
        
        try:
            self.pipeline = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def evaluate_model(self, X_test, y_test):
        if self.pipeline is None:
            raise RuntimeError("Model not trained")
        
        y_pred = self.pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return {
            "accuracy": float(accuracy),
            "f1_score": float(f1)
        }
    

    def save_metadata(self, metrics):
        """Сохраняет метаданные модели в JSON"""
        metadata = {
            "trained": True,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics
        }
        
        # Создаём папку если нет
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем в JSON
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        # Обновляем в памяти
        self.metadata = metadata

    def load_metadata(self):
        """Загружает метаданные из JSON"""
        if not self.metadata_path.exists():
            self.metadata = None
            return
        
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load metadata: {e}")
            self.metadata = None

    def get_status(self):
        """Возвращает статус модели"""
        # Если метаданных нет - модель не обучена
        if self.metadata is None:
            return {
                "trained": False,
                "trained_at": None,
                "metrics": None
            }
        
        # Если есть - возвращаем
        return self.metadata
    
    def predict(self, features):
        if self.pipeline is None:
            raise RuntimeError("Model not trained")
        
        # Pydantic → dict → DataFrame
        data = [features.model_dump()]
        df = pd.DataFrame(data)
        
        # Предсказание
        prediction = self.pipeline.predict(df)[0]
        probabilities = self.pipeline.predict_proba(df)[0]
        
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probabilities[1]),
            "probabilities": [float(p) for p in probabilities]
        }
