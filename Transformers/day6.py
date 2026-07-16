import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW  # Импортируем оптимизатор из PyTorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Настройки
model_name = "distilbert-base-multilingual-cased"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используем устройство: {device}")

# Сначала создадим фейковый датасет для теста, чтобы код гарантированно работал
if not os.path.exists('your_dataset.csv'):
    demo_df = pd.DataFrame({
        'text': [
            "I love this product, it works perfectly!",
            "Terrible experience, very bad quality.",
            "Normal quality, not great but okay.",
            "Absolutely amazing, highly recommend!",
            "Trash, waste of money and time.",
            "Decent product, does its job.",
            "Worst purchase ever, so disappointed.",
            "Excellent support and fast delivery!"
        ],
        'label': [1, 0, 1, 1, 0, 1, 0, 1]  # 0 - Negative, 1 - Positive
    })
    demo_df.to_csv('your_dataset.csv', index=False)

# ==========================================
# ЗАДАЧА 1: Подготовка Dataset класса
# ==========================================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Токенизируем отдельный текст
        encoding = self.tokenizer(
            text,
            padding='max_length',  # Добиваем все тексты до max_length
            truncation=True,       # Обрезаем если текст длиннее
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# ЗАДАЧА 2: Загрузка и подготовка данных
# ==========================================
# 1. Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Загружаем датасет
df = pd.read_csv('your_dataset.csv')
texts = df['text'].tolist()
labels = df['label'].tolist()

# 3. Разделяем на train/validation (80/20) с сохранением пропорций классов (stratify)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 4. Создаем PyTorch Dataset объекты
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)


# ==========================================
# ЗАДАЧА 3: Загрузка модели для классификации
# ==========================================
# Определяем количество уникальных классов
num_labels = len(set(labels)) 

# Загружаем модель со специальной "головой" для классификации поверх BERT
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# Создаем DataLoaders (нарезка данных на батчи)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)


# ==========================================
# ЗАДАЧА 4: Настройка обучения
# ==========================================
# Оптимизатор AdamW из PyTorch (без аргумента correct_bias)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Переносим модель на GPU (если есть) или на CPU
model.to(device)


# ==========================================
# ЗАДАЧА 5: Функция для обучения одной эпохи
# ==========================================
def train_epoch(model, dataloader, optimizer, device):
    model.train() # Важно! Переводим модель в режим обучения
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad() # Сбрасываем старые градиенты

        # Переносим тензоры на видеокарту/процессор
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Прямой проход (Forward pass). Модель сама считает Loss, если передать labels
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        
        # Обратный проход (Backward pass) — вычисляем градиенты
        loss.backward()
        
        # Шаг оптимизатора — обновляем веса модели
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ==========================================
# ЗАДАЧА 6: Функция для оценки (Валидация)
# ==========================================
def evaluate(model, dataloader, device):
    model.eval() # Переводим модель в режим предсказания (выключает dropout)
    predictions = []
    true_labels = []

    # Отключаем расчет градиентов для экономии памяти и ускорения вычислений
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Берём индекс класса с наибольшей вероятностью из логитов
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')

    return accuracy, f1


# ==========================================
# ЗАДАЧА 7: Обучение модели
# ==========================================
num_epochs = 3
print("\n=== Старт обучения ===")

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_acc, val_f1 = evaluate(model, val_loader, device)

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Accuracy: {val_acc:.4f}')
    print(f'Val F1: {val_f1:.4f}')
    print('-' * 50)


# ==========================================
# ЗАДАЧА 8: Сохранение модели и метрик
# ==========================================
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

with open('fine_tuned_results.txt', 'w', encoding='utf-8') as f:
    f.write(f'Final Validation F1: {val_f1:.4f}\n')
    f.write(f'Final Validation Accuracy: {val_acc:.4f}\n')

print("\nМодель и результаты успешно сохранены!")