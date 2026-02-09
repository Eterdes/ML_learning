import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class ModelNotTrainedError(RuntimeError):
    pass


class ModelService:
    MODEL_CLASSES = {
        "logreg": LogisticRegression,
        "random_forest": RandomForestClassifier,
    }

    NUMERIC_FEATURES = [
        "monthly_fee",
        "usage_hours",
        "support_requests",
        "account_age_months",
        "failed_payments",
        "autopay_enabled",
    ]

    CATEGORICAL_FEATURES = [
        "region",
        "device_type",
        "payment_method",
    ]

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.pipeline = None
        self.metadata = {}

    # ---------- helpers ----------
    def _all_features(self) -> list[str]:
        return self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES

    def get_model_by_type(self, model_type: str, hyperparameters: dict | None):
        if hyperparameters is None:
            hyperparameters = {}

        if model_type not in self.MODEL_CLASSES:
            raise ValueError(
                f"Unknown model_type: {model_type}. Available: {list(self.MODEL_CLASSES.keys())}"
            )

        model_class = self.MODEL_CLASSES[model_type]
        try:
            return model_class(**hyperparameters)
        except TypeError as e:
            # неправильные гиперпараметры -> 400
            raise ValueError(f"Invalid hyperparameters for {model_type}: {str(e)}")

    # ---------- train / eval ----------
    def train_churn_model(self, X_train, y_train, model_type: str = "logreg", hyperparameters: dict | None = None):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.CATEGORICAL_FEATURES),
            ]
        )

        model = self.get_model_by_type(model_type, hyperparameters)

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        # если X_train не содержит нужные колонки -> будет ошибка, пусть уйдет как ValueError (400)
        self.pipeline.fit(X_train, y_train)

        self.metadata["model_type"] = model_type
        self.metadata["hyperparameters"] = hyperparameters or {}


    def evaluate_model(self, X_test, y_test) -> dict:
        if self.pipeline is None:
            raise ModelNotTrainedError("Model not trained")

        y_pred = self.pipeline.predict(X_test)

        roc_auc = None
        # roc_auc считаем по вероятностям и только если есть оба класса в y_test
        if len(set(y_test)) == 2:
            y_proba = self.pipeline.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": roc_auc
        }
    

    # ---------- persistence ----------
    def save_churn_model(self):
        if self.pipeline is None:
            raise ModelNotTrainedError("Model not trained")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)

    def load_churn_model(self):
        if self.model_path.exists():
            self.pipeline = joblib.load(self.model_path)

    def save_metadata(self, metrics: dict):
        metadata_path = self.model_path.parent / "model_metadata.json"
        self.metadata.update({"metrics": metrics})
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def load_metadata(self):
        metadata_path = self.model_path.parent / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def get_status(self):
        return {
            "model_loaded": self.pipeline is not None,
            "model_type": self.metadata.get("model_type", "unknown"),
            "hyperparameters": self.metadata.get("hyperparameters", {}),
            "metrics": self.metadata.get("metrics", {}),
        }

    # ---------- predict ----------
    def predict(self, features) -> dict:
        """
        features: FeatureVectorChurn (pydantic model)
        """
        if self.pipeline is None:
            raise ModelNotTrainedError("Model not trained. Train model first using /model/train")

        all_features = self._all_features()

        # pydantic v2
        payload = features.model_dump()

        # защита от "неверное количество признаков" (на случай, если кто-то подменит модель/вызов)
        missing = [f for f in all_features if f not in payload]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        df = pd.DataFrame([payload])[all_features]

        pred = self.pipeline.predict(df)[0]
        proba = self.pipeline.predict_proba(df)[0][1]

        return {
            "churn_prediction": int(pred),
            "churn_probability": float(proba),
        }

    def get_feature_schema(self):
        feature_types = {}

        for f in self.NUMERIC_FEATURES:
            if ("requests" in f) or ("age" in f) or ("payments" in f) or ("autopay" in f):
                feature_types[f] = "int"
            else:
                feature_types[f] = "float"

        for f in self.CATEGORICAL_FEATURES:
            feature_types[f] = "str"

        return {
            "numeric_features": self.NUMERIC_FEATURES,
            "categorical_features": self.CATEGORICAL_FEATURES,
            "feature_types": feature_types,
            "total_features": len(self.NUMERIC_FEATURES) + len(self.CATEGORICAL_FEATURES),
        }
