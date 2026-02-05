import pandas as pd
from pathlib import Path
from typing import Any
from sklearn.model_selection import train_test_split


class DatasetService:
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.df: pd.DataFrame | None = None

    def load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        if self.df.empty:
            raise ValueError("Dataset is empty")

    def preview(self, n: int = 10) -> list[dict[str, Any]]:
        if self.df is None:
            raise RuntimeError("Dataset is not loaded. Call load() first.")

        n = max(1, min(int(n), 100))  # ограничим N, чтобы не возвращать слишком много
        rows = self.df.head(n).to_dict(orient="records")
        return rows

    def info(self) -> dict[str, Any]:
        if self.df is None:
            raise RuntimeError("Dataset is not loaded. Call load() first.")

        rows_count, cols_count = self.df.shape
        columns = list(self.df.columns)

        if "churn" not in self.df.columns:
            raise ValueError("Column 'churn' not found in dataset")

        churn_counts = self.df["churn"].value_counts(dropna=False).to_dict()
        # Приведём ключи к строкам (JSON-friendly)
        churn_distribution = {str(k): int(v) for k, v in churn_counts.items()}

        return {
            "rows": int(rows_count),
            "cols": int(cols_count),
            "columns": columns,
            "churn_distribution": churn_distribution,
        }
    
    # day 4 
    def split_data(self) -> None:
        if self.df is None:
            raise RuntimeError("Dataset is not loaded. Call load() first.")
        
        y = self.df["churn"]
        X = self.df.drop(columns=["churn"])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_split_info(self) -> dict:
        if self.X_train is None:
            raise RuntimeError("Data is not split. Call split_data() first.")
        
        train_size = len(self.X_train)
        test_size = len(self.X_test)

        train_distribution = self.y_train.value_counts().to_dict()
        train_churn = {str(k): int(v) for k, v in train_distribution.items()}

        test_distribution = self.y_test.value_counts().to_dict()
        test_churn = {str(k): int(v) for k, v in test_distribution.items()}

        return {
        "train_size": train_size,
        "test_size": test_size,
        "train_churn_distribution": train_churn,
        "test_churn_distribution": test_churn
        }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "churn_dataset.csv"

    service = DatasetService(csv_path)
    service.load()

    print("Preview(3):")
    print(service.preview(3))

    print("\nInfo:")
    print(service.info())