FROM python:3.12-slim

WORKDIR /app

# быстрее и чище логи
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ставим зависимости
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi uvicorn[standard] pandas scikit-learn joblib

# копируем код
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "api.fastapi_churn_day13:app", "--host", "0.0.0.0", "--port", "8000"]
