import os

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# Настройки отказоустойчивости
MAX_RETRIES = 3
RETRY_DELAY = 0.5  # в секундах