import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Находим путь к папке, в которой лежит этот скрипт, 
# чтобы графики и файлы сохранялись строго рядом со скриптом
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else ''
file_path = os.path.join(script_dir, 'financial_data.csv') if script_dir else 'financial_data.csv'

print("--- ЧАСТЬ 1: Загрузка и EDA ---")

# 1. Загружаем файл без заголовков и даем колонкам имена
df = pd.read_csv(file_path, header=None, names=['sentiment', 'text'])
df['sentiment'] = df['sentiment'].str.strip()

print("Размерность датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

# Проверяем и удаляем пропуски в текстах
print("\nПропущенные значения:\n", df.isnull().sum())
df = df.dropna(subset=['text'])

# Настройка стиля seaborn для графиков
sns.set_theme(style="whitegrid")

# Построение графика распределения классов
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='sentiment', hue='sentiment', palette='viridis', legend=False)
plt.title('Распределение классов тональности')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.tight_layout()

# Сохраняем график прямо в папку Classification
plot1_path = os.path.join(script_dir, 'sentiment_distribution.png') if script_dir else 'sentiment_distribution.png'
plt.savefig(plot1_path)
plt.close()
print(f"\n[Успех] График распределения классов сохранен в: {plot1_path}")

# 1.2 Анализ длин текстов
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

# Гистограмма длин текстов
plt.figure(figsize=(10, 5))
df['text_length'].hist(bins=50, color='skyblue', edgecolor='black')
plt.title('Распределение длин текстов')
plt.xlabel('Длина текста (в символах)')
plt.ylabel('Количество')
plt.tight_layout()

plot2_path = os.path.join(script_dir, 'text_length_distribution.png') if script_dir else 'text_length_distribution.png'
plt.savefig(plot2_path)
plt.close()
print(f"[Успех] График распределения длин сохранен в: {plot2_path}")

# Вывод примеров для каждого класса
print("\n--- Примеры текстов для каждого класса ---")
for sentiment in df['sentiment'].unique():
    print(f'\n=== Класс {sentiment} ===')
    subset = df[df['sentiment'] == sentiment]
    if not subset.empty:
        print(subset['text'].iloc[0])


print("\n--- ЧАСТЬ 2: Подготовка данных ---")

# 2.1 Очистка текста
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text) # Удаляем HTML-теги, если они есть
    text = re.sub(r'\s+', ' ', text)     # Схлопываем множественные пробелы
    text = text.strip()
    return text

df['text_clean'] = df['text'].apply(clean_text)

# Удаляем пустые тексты после очистки
df = df[df['text_clean'].str.len() > 0]

# 2.2 Добавление дополнительных признаков
df['word_count_clean'] = df['text_clean'].str.split().str.len()
df['char_count'] = df['text_clean'].str.len()
# Считаем знаки доллара (часто встречаются в финансовых новостях)
df['dollar_count'] = df['text_clean'].apply(lambda x: x.count('$'))

# Преобразуем текстовые метки классов в цифры (0, 1, 2)
df['label'] = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})

print("Уникальные числовые метки (label):", df['label'].unique())
print("Итоговая размерность датасета после очистки:", df.shape)

print("\n--- ЧАСТЬ 3: Baseline модель ---")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# 1. Создаем TF-IDF векторизатор
# max_features=5000 берет только 5000 самых важных слов
# ngram_range=(1, 2) заставляет модель видеть и отдельные слова ("profit"), и пары слов ("high profit")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# 2. Превращаем тексты в огромную матрицу чисел
X = vectorizer.fit_transform(df['text_clean'])
y = df['label'].values

# 3. Делим данные на обучающие (80%) и тестовые (20%)
# stratify=y гарантирует, что в обеих частях будет одинаковый процент хороших/плохих новостей
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Инициализируем и обучаем Логистическую регрессию
model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
print("Обучаем модель...")
model.fit(X_train, y_train)

# 5. Просим модель предсказать тональность для новостей, которые она ЕЩЕ НЕ ВИДЕЛА (X_test)
y_pred = model.predict(X_test)

# 6. Выводим подробный отчет о качестве работы
print("\nОтчет классификации (Classification Report):")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

# Считаем главную метрику — Macro F1
f1 = f1_score(y_test, y_pred, average='macro')
print(f'Baseline Macro F1: {f1:.4f}')

# 7. Сохраняем baseline результат в файл
baseline_path = os.path.join(script_dir, 'baseline_result.txt') if script_dir else 'baseline_result.txt'
with open(baseline_path, 'w') as f:
    f.write(f'Baseline Macro F1: {f1:.4f}')

print(f"\n[Успех] Результаты baseline сохранены в файл: {baseline_path}")

print("\n--- ЧАСТЬ 4: Улучшение модели ---")
import spacy
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import joblib

# Загружаем spaCy для лемматизации
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Используем nlp.pipe для быстрой пакетной обработки текстов
print("Запуск лемматизации (это может занять около 1-2 минут)...")
docs = list(nlp.pipe(df['text_clean'], batch_size=1000))
df['text_lemm'] = [' '.join([token.lemma_ for token in doc]) for doc in docs]
print("[Успех] Лемматизация завершена!")

# Словарь для хранения результатов всех экспериментов
experiment_results = {}

# --- ПОДХОД 1: Лемматизация + Baseline TF-IDF + LogReg ---
print("\n[Эксперимент 1] Обучение LogReg на лемматизированных текстах...")
vectorizer_l = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_lemm = vectorizer_l.fit_transform(df['text_lemm'])

# Делим лемматизированные данные строго с тем же random_state
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_lemm, y, test_size=0.2, random_state=42, stratify=y
)

model_l = LogisticRegression(max_iter=1000, random_state=42)
model_l.fit(X_train_l, y_train_l)
pred_l = model_l.predict(X_test_l)
f1_exp1 = f1_score(y_test_l, pred_l, average='macro')
experiment_results['Лемматизация + LogReg'] = f1_exp1
print(f"F1 для Эксперимента 1: {f1_exp1:.4f}")


# --- ПОДХОД 2: Лемматизация + Тройные n-граммы (1, 3) + LinearSVC ---
print("\n[Эксперимент 2] Обучение LinearSVC с триграммами...")
vectorizer_tri = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X_tri = vectorizer_tri.fit_transform(df['text_lemm'])

X_train_tri, X_test_tri, y_train_tri, y_test_tri = train_test_split(
    X_tri, y, test_size=0.2, random_state=42, stratify=y
)

model_svc = LinearSVC(max_iter=5000, random_state=42)
model_svc.fit(X_train_tri, y_train_tri)
pred_svc = model_svc.predict(X_test_tri)
f1_exp2 = f1_score(y_test_tri, pred_svc, average='macro')
experiment_results['Лемматизация + Tri-grams + LinearSVC'] = f1_exp2
print(f"F1 для Эксперимента 2: {f1_exp2:.4f}")


# --- ПОДХОД 3: Подбор гиперпараметров (GridSearchCV) для LinearSVC ---
print("\n[Эксперимент 3] Подбор гиперпараметра C для LinearSVC...")
param_grid = {'C': [0.1, 0.5, 1.0, 5.0]}
# cv=3 означает кросс-валидацию по 3 фолдам
grid = GridSearchCV(LinearSVC(max_iter=5000, random_state=42), param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
grid.fit(X_train_tri, y_train_tri)

best_svc = grid.best_estimator_
pred_grid = best_svc.predict(X_test_tri)
f1_exp3 = f1_score(y_test_tri, pred_grid, average='macro')
experiment_results['Лемматизация + Tri-grams + Tuned LinearSVC'] = f1_exp3
print(f"Лучший параметр C: {grid.best_params_}")
print(f"F1 для Эксперимента 3: {f1_exp3:.4f}")


# Сохраняем лучшую модель и векторизатор на диск
best_model_path = os.path.join(script_dir, 'best_model.pkl')
best_vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')
joblib.dump(best_svc, best_model_path)
joblib.dump(vectorizer_tri, best_vectorizer_path)
print(f"\n[Успех] Лучшая модель сохранена в: {best_model_path}")


print("\n--- ЧАСТЬ 5: Сравнение и анализ ---")
# 5.1 Генерация markdown-таблицы результатов
with open(os.path.join(script_dir, 'comparison_table.txt'), 'w', encoding='utf-8') as f:
    f.write("| Метод | Macro F1 | Улучшение |\n")
    f.write("| :--- | :--- | :--- |\n")
    f.write(f"| Baseline TF-IDF + LogReg | {f1:.4f} | - |\n")
    for method, f1_score_val in experiment_results.items():
        improvement = ((f1_score_val - f1) / f1) * 100
        f.write(f"| {method} | {f1_score_val:.4f} | {improvement:+.2f}% |\n")
print("[Успех] Таблица результатов сохранена в 'comparison_table.txt'")

# 5.2 Матрица ошибок (Confusion Matrix) для лучшей модели
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_tri, pred_grid)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Матрица ошибок лучшей модели (Tuned LinearSVC)')
plt.ylabel('Истинные классы')
plt.xlabel('Предсказанные классы')
plt.tight_layout()

cm_path = os.path.join(script_dir, 'confusion_matrix.png')
plt.savefig(cm_path)
plt.close()
print(f"[Успех] Матрица ошибок сохранена в: {cm_path}")

# 5.3 Анализ ошибок в текстовый файл
# Восстановим индексы тестовой выборки, чтобы сопоставить предсказания с исходным текстом
_, test_indices = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)
test_df = df.loc[test_indices].copy()
test_df['predicted'] = pred_grid

errors = test_df[test_df['label'] != test_df['predicted']]

error_path = os.path.join(script_dir, 'error_analysis.txt')
with open(error_path, 'w', encoding='utf-8') as f:
    f.write("=== АНАЛИЗ ОШИБОК ЛУЧШЕЙ МОДЕЛИ ===\n\n")
    f.write(f"Всего ошибок на тестовой выборке: {len(errors)} из {len(test_df)}\n\n")
    for idx, row in errors.head(10).iterrows():
        f.write(f"Исходный текст: {row['text']}\n")
        f.write(f"Истинный класс: {row['sentiment']} ({row['label']})\n")
        f.write(f"Предсказанный класс: {row['predicted']}\n")
        f.write("-" * 50 + "\n")
print(f"[Успех] Анализ ошибок сохранен в: {error_path}")


print("\n--- ЧАСТЬ 6: Предсказание на новых данных ---")
def predict_sentiment(texts, model, vectorizer):
    if isinstance(texts, str):
        texts = [texts]
    
    # Предобработка новых текстов
    cleaned = [clean_text(t) for t in texts]
    lemmed = [' '.join([token.lemma_ for token in nlp(t)]) for t in cleaned]
    
    # Векторизация и предсказание
    X_new = vectorizer.transform(lemmed)
    predictions = model.predict(X_new)
    
    reverse_mapping = {0: 'Negative 🔴', 1: 'Neutral 🟡', 2: 'Positive 🟢'}
    return [reverse_mapping[p] for p in predictions]

# 5 тестовых примеров
custom_news = [
    "Apple quarterly profits beat all Wall Street expectations, stock surges 8%.",
    "Company announces bankruptcy and immediate closure of all retail stores.",
    "Gold prices remain stable as investors await the upcoming Fed meeting.",
    "Microsoft announces major layoffs affecting over five thousand employees.",
    "The corporation reports steady revenue growth in the third quarter."
]

results = predict_sentiment(custom_news, best_svc, vectorizer_tri)

for news, label in zip(custom_news, results):
    print(f"\nНовость: {news}")
    print(f"Предсказание модели: {label}")