import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(
    title="Sentiment Analysis API",
    description="Продакшн API сервис для определения тональности текстов на базе Fine-tuned BERT.",
    version="1.0.0"
)

# Загружаем модель глобально при старте приложения
model_path = './fine_tuned_model'
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_ft = AutoModelForSequenceClassification.from_pretrained(model_path)
    model_ft.eval()
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить обученную модель из {model_path}. Ошибка: {e}")

label_map = {0: 'Negative', 1: 'Positive'}

# Описываем схему входящего JSON-запроса через Pydantic
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст отзыва для анализа тональности")

# Описываем схему исходящего ответа
class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model_ft(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        
        # Готовим словарь вероятностей
        probabilities_dict = {
            label_map.get(i, f"Class_{i}"): float(prob)
            for i, prob in enumerate(probs)
        }
        
        return PredictionResponse(
            prediction=label_map.get(pred, str(pred)),
            probabilities=probabilities_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Запускаем сервер на порту 8000
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)