import json
import logging
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.pipeline import LLMPipeline

# Настройка логов
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("App")

pipeline = LLMPipeline()

# Инициализация FastAPI
app = FastAPI(title="LLM Customer Support Pipeline API", version="1.0.0")

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    category: str
    sentiment: str
    summary: str
    final_answer: str

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    try:
        result = pipeline.process(request.text)
        return result
    except Exception as e:
        logger.error(f"Ошибка API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_local_demo():
    logger.info("=== Запуск локального демо-тестирования ===")
    try:
        with open("demo_inputs.json", "r", encoding="utf-8") as f:
            demo_data = json.load(f)
    except FileNotFoundError:
        logger.error("Файл demo_inputs.json не найден!")
        return

    results = []
    for idx, item in enumerate(demo_data, 1):
        text = item.get("text", "")
        print(f"\n--- Тест #{idx} ---")
        print(f"Клиент: {text}")
        
        res = pipeline.process(text)
        
        print(f"Категория: {res['category'].upper()}")
        print(f"Тональность: {res['sentiment'].upper()}")
        print(f"Сводка: {res['summary']}")
        print(f"Ответ ИИ:\n{res['final_answer']}")
        print("-" * 50)
        results.append(res)

    with open("demo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info("Демо-тесты завершены. Результаты сохранены в demo_results.json")

if __name__ == "__main__":
    # Если запущен напрямую — прогоняем демо
    run_local_demo()