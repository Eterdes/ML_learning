import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Инициализация модели и токенизатора из сохраненной папки
model_path = './fine_tuned_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model_ft = AutoModelForSequenceClassification.from_pretrained(model_path)
model_ft.eval()

# Определяем динамический маппинг классов
# Если классов 2, пишем: {0: 'Negative', 1: 'Positive'}
# Если 3, пишем: {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
label_map = {0: 'Negative', 1: 'Positive'} 

def predict_sentiment(text):
    if not text.strip():
        return "Пожалуйста, введите текст."
        
    # Токенизируем входящую строку
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model_ft(**inputs)
        
    # Считаем вероятности классов
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()
    
    # Форматируем структурированный вывод
    result = f"🎯 Вердикт модели: {label_map.get(pred, f'Класс {pred}')}\n\n"
    result += "📊 Вероятности по классам:\n"
    for i, prob in enumerate(probs):
        class_name = label_map.get(i, f"Класс {i}")
        result += f"  • {class_name}: {prob.item() * 100:.2f}%\n"
        
    return result

# Строим веб-интерфейс
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Введите отзыв для анализа... например: I loved this film!"),
    outputs=gr.Textbox(label="Результат анализа"),
    title="Sentiment Analysis Classifier (BERT)",
    description="Интерфейс для демонстрации работы вашей fine-tuned модели классификации тональности текста.",
    theme="soft"
)

if __name__ == "__main__":
    # Запуск локального сервера
    demo.launch()