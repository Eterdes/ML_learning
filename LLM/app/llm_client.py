from openai import OpenAI
from app.config import OLLAMA_API_BASE, OLLAMA_MODEL

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=OLLAMA_API_BASE,
            api_key="ollama"  # заглушка для локального запуска
        )
        self.model = OLLAMA_MODEL

    def get_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content