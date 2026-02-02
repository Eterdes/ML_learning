import pandas as pd
from pathlib import Path
from typing import Any


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


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "churn_dataset.csv"

    service = DatasetService(csv_path)
    service.load()

    print("Preview(3):")
    print(service.preview(3))

    print("\nInfo:")
    print(service.info())