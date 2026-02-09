import numpy as np
from final_api.services.dataset_service import DatasetService

def test_dataset_load_info_preview_split(synth_csv):
    svc = DatasetService(synth_csv)

    svc.load()
    info = svc.info()

    assert info["rows"] == 200
    assert "churn" in info["columns"]
    assert set(info["churn_distribution"].keys()) <= {"0", "1"}

    prev = svc.preview(5)
    assert isinstance(prev, list)
    assert len(prev) == 5
    assert "monthly_fee" in prev[0]

    # воспроизводимость разбиения (у тебя random_state не задан)
    np.random.seed(42)
    svc.split_data()

    split = svc.get_split_info()
    assert split["train_size"] == 160
    assert split["test_size"] == 40
