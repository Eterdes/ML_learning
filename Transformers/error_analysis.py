import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split

# Настройки
model_path = './fine_tuned_model'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Сначала обучи и сохрани модель в папку {model_path} (День 5)!")

# 1. Загружаем токенизатор и дообученную модель
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

# 2. Восстанавливаем тест-выборку строго так же, как в Днях 5 и 6
df = pd.read_csv('your_dataset.csv')
_, test_df, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=42, stratify=df['label'])

test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

# 3. Делаем предсказания (Batch Inference)
y_pred_ft = []
batch_size = 16

for i in range(0, len(test_texts), batch_size):
    batch_texts = test_texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    y_pred_ft.extend(preds)

# ==========================================
# ЗАДАЧА 1 и 2: Выделение FP / FN и паттернов
# ==========================================
df_test = pd.DataFrame({
    'text': test_texts,
    'true_label': test_labels,
    'pred_label': y_pred_ft
})

# Ошибки — там где метки не совпали
errors = df_test[df_test['true_label'] != df_test['pred_label']].copy()

# False Positives: модель думает, что всё круто (1), а на самом деле всё плохо (0)
fp = errors[(errors['pred_label'] == 1) & (errors['true_label'] == 0)]

# False Negatives: модель думает, что всё плохо (0), а на самом деле всё круто (1)
fn = errors[(errors['pred_label'] == 0) & (errors['true_label'] == 1)]

print(f"Всего тестовых примеров: {len(df_test)}")
print(f"Всего ошибок: {len(errors)}")
print(f"False Positives (Ложные тревоги): {len(fp)}")
print(f"False Negatives (Пропуски целей): {len(fn)}")

# Считаем длину текстов
errors['text_length'] = errors['text'].str.len()
avg_err_len = errors['text_length'].mean() if len(errors) > 0 else 0
avg_all_len = df_test['text'].str.len().mean()

print(f"\nСредняя длина ошибочных текстов: {avg_err_len:.0f} символов")
print(f"Средняя длина всех текстов: {avg_all_len:.0f} символов")

# Записываем глубокий анализ ошибок в файл
with open('error_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('=== СИСТЕМНЫЙ АНАЛИЗ ОШИБОК ===\n\n')
    f.write(f'Всего тестовых примеров: {len(df_test)}\n')
    f.write(f'Всего ошибок: {len(errors)}\n')
    f.write(f'False Positives: {len(fp)}\n')
    f.write(f'False Negatives: {len(fn)}\n\n')
    f.write(f'Средняя длина ошибочных текстов: {avg_err_len:.0f} симв.\n')
    f.write(f'Средняя длина всех текстов: {avg_all_len:.0f} симв.\n\n')
    
    f.write('=== ПРИМЕРЫ FALSE POSITIVES (Ожидали Bad, модель выдала Good) ===\n')
    for idx, row in fp.head(5).iterrows():
        f.write(f'\nТекст: {row["text"]}\n')
        f.write(f'Истинный: {row["true_label"]}, Предсказан: {row["pred_label"]}\n')
        
    f.write('\n\n=== ПРИМЕРЫ FALSE NEGATIVES (Ожидали Good, модель выдала Bad) ===\n')
    for idx, row in fn.head(5).iterrows():
        f.write(f'\nТекст: {row["text"]}\n')
        f.write(f'Истинный: {row["true_label"]}, Предсказан: {row["pred_label"]}\n')
        
    f.write('\n\n=== НАБЛЮДЕНИЯ И ВЫВОДЫ ===\n')
    if avg_err_len > avg_all_len * 1.3:
        f.write("- Ошибки чаще происходят на длинных текстах. Возможно, модель теряет контекст из-за обрезки (truncation) или рассеивания внимания на последних слоях.\n")
    else:
        f.write("- Длина текста не оказывает сильного влияния на ошибки. Проблема, скорее всего, в семантике.\n")
    f.write("- Модель может путаться в текстах с сарказмом, двойным отрицанием или сложными речевыми конструкциями.\n")

print("\nАнализ сохранен в 'error_analysis.txt'")