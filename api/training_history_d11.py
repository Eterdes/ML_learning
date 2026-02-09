import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class TrainingHistoryService:
    def __init__(self, history_path: Path):
        self.history_path = history_path
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.history_path.exists():
            self._save([])

    def _load(self) -> list[dict]:
        with open(self.history_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: list[dict]) -> None:
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_record(
        self,
        model_type: str,
        hyperparameters: dict,
        metrics: dict,
    ) -> None:
        history = self._load()

        history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
        })

        self._save(history)

    def get_history(
        self,
        model_type: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        history = self._load()

        if model_type:
            history = [h for h in history if h["model_type"] == model_type]

        return history[-limit:]