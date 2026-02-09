# Churn Prediction Service

ML-сервис для прогнозирования оттока клиентов (churn) на основе пользовательских характеристик.
Реализован как REST API на FastAPI с возможностью обучения модели, получения предсказаний,
мониторинга состояния сервиса и контейнеризации через Docker.

---

## Цель сервиса

Сервис решает задачу бинарной классификации:
**уйдёт ли клиент (churn = 1) или останется (churn = 0)**.

Функциональность:
- загрузка и подготовка churn-датасета
- обучение ML-модели через API
- получение предсказаний по одному или нескольким клиентам
- просмотр статуса модели и истории обучения
- health-check для эксплуатации
- запуск локально и в Docker

---

## Структура проекта

```text
final_api/
├── main.py # точка входа FastAPI
│
├── data/
│ └── churn_dataset.csv # исходный датасет
│
├── models/
│ ├── churn_model.joblib # сохранённая ML-модель
│ ├── model_metadata.json # метаданные модели (метрики, фичи)
│ └── training_history.json # история обучений
│
├── services/
│ ├── dataset_service.py # загрузка и подготовка датасета
│ ├── model_service.py # обучение, инференс, сохранение модели
│ └── training_history.py # логирование истории обучений
│
├── schemas/
│ └── churn.py # Pydantic-схемы (request/response)
│
├── tests/
│ ├── conftest.py
│ ├── test_api_integration.py # e2e тесты FastAPI
│ ├── test_dataset_service.py # unit-тесты датасета
│ └── test_model_service_unit.py # unit-тесты модели
│
└── init.py # (присутствует во всех пакетах)
```

---

## Формат датасета churn_dataset.csv

Расположение:
```
final_api/data/churn_dataset.csv
```

Формат:
- CSV
- с заголовком
- разделитель `,`

### Колонки датасета

| Колонка | Тип | Пример | Описание |
|------|-----|-------|---------|
| `monthly_fee` | float | `19.99` | ежемесячная плата |
| `usage_hours` | float | `27.92` | часы использования сервиса |
| `support_requests` | int | `2` | обращения в поддержку |
| `account_age_months` | int | `14` | возраст аккаунта |
| `failed_payments` | int | `1` | неудачные платежи |
| `region` | str | `europe` | регион |
| `device_type` | str | `mobile` | тип устройства |
| `payment_method` | str | `card` | метод оплаты |
| `autopay_enabled` | int (0/1) | `1` | автоплатёж |
| `churn` | int (0/1) | `0` | **таргет** |

### Пример строк CSV

```csv
monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
9.99,27.92,1,14,1,america,desktop,card,1,1
19.99,21.48,2,1,0,europe,mobile,paypal,0,0 
```

Поле churn обязательно для обучения,
не передаётся при /predict.

---

## Запуск локально

### Установка зависимостей

```bash
py -m pip install -r requirements.txt
```

### Минимально
```bash
py -m pip install fastapi uvicorn[standard] pandas scikit-learn joblib pytest httpx
```

### Запуск сервиса локально
```bash
py -m uvicorn final_api.main:app --reload
```

При старте загружается датасет, выполняется split, загружается модель (если существует).

### Проверка
Swagger: http://127.0.0.1:8000/docs

Healthcheck: http://127.0.0.1:8000/health

Пример ответа /health:
```json
{
  "status": "ok",
  "dataset_loaded": true,
  "split_ready": true,
  "model_loaded": true
}
```

---

## Запуск в Docker

Сборка образа
```bash
docker build -t churn-service .
```

Запуск контейнера
```bash
docker run --rm -p 8000:8000 churn-service
```

---

## Примеры запросов

POST /model/train
```json
{
  "model_type": "logreg",
  "hyperparameters": {}
}
```

Ответ
```json
{
  "status": "trained",
  "model_type": "logreg",
  "hyperparameters": {},
  "metrics": {
    "accuracy": 0.80,
    "precision": 0.67,
    "recall": 0.02,
    "f1": 0.05,
    "roc_auc": 0.64
  }
}
```

POST /predict
```json
{
  "monthly_fee": 9.99,
  "usage_hours": 12.3,
  "support_requests": 1,
  "account_age_months": 10,
  "failed_payments": 0,
  "region": "europe",
  "device_type": "mobile",
  "payment_method": "card",
  "autopay_enabled": 1
}
```

Ответ
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.23
}
```

---

## Тестирование

```bash
py -m pytest -q
```

Включает unit-тесты сервисов, интеграционные тесты FastAPI и проверки edge-cases.

---

## Контроль версий

Финальное состояние проекта зафиксировано в системе контроля версий Git:

```bash
git add .
git commit -m "Final version: churn prediction service"
```

---

## Итог
Проект включает полный ML-pipeline с корректной архитектурой, логированием, healthcheck, тестами и Docker-готовностью. Готов к использованию и демонстрации.