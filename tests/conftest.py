import numpy as np
import pandas as pd
import pytest
from pathlib import Path

def make_synth_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "monthly_fee": rng.uniform(5, 50, size=n).round(2),
        "usage_hours": rng.uniform(0, 100, size=n).round(2),
        "support_requests": rng.integers(0, 6, size=n),
        "account_age_months": rng.integers(0, 36, size=n),
        "failed_payments": rng.integers(0, 4, size=n),
        "region": rng.choice(["europe", "america"], size=n),
        "device_type": rng.choice(["mobile", "desktop"], size=n),
        "payment_method": rng.choice(["card", "paypal"], size=n),
        "autopay_enabled": rng.integers(0, 2, size=n),
    })

    # делаем churn зависимым от пары факторов, чтобы модель точно чему-то училась
    score = (
        (df["monthly_fee"] > 30).astype(int)
        + (df["failed_payments"] > 1).astype(int)
        + (df["support_requests"] > 2).astype(int)
        - (df["autopay_enabled"] == 1).astype(int)
    )
    # нормируем, чтобы оба класса были
    df["churn"] = (score > 1).astype(int)

    # гарантируем оба класса (на всякий)
    if df["churn"].nunique() < 2:
        df.loc[: n//2, "churn"] = 0
        df.loc[n//2 :, "churn"] = 1

    return df

@pytest.fixture
def synth_csv(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "churn_dataset.csv"
    make_synth_df(200).to_csv(csv_path, index=False)
    return csv_path