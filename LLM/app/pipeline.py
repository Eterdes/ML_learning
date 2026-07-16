import json
import logging
import time
from app.config import MAX_RETRIES, RETRY_DELAY
from app.llm_client import LLMClient
from app.prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFIER_USER_TEMPLATE,
    ROUTING_PROMPTS,
    SELF_CHECK_PROMPT_TEMPLATE
)
from app.schemas import ClassificationSchema

logger = logging.getLogger("ResilientPipeline")

class LLMPipeline:
    def __init__(self):
        self.client = LLMClient()

    def _get_completion_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.get_completion(system_prompt, user_prompt)
                if not response or not response.strip():
                    raise ValueError("Пустой ответ от LLM")
                return response
            except Exception as e:
                logger.warning(f"Попытка {attempt}/{MAX_RETRIES} провалилась: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("Все попытки запроса к LLM исчерпаны.")
                    raise e

    def _clean_and_parse_json(self, raw_text: str) -> dict:
        try:
            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}")
            if start_idx == -1 or end_idx == -1:
                raise ValueError("JSON скобки не найдены")
            return json.loads(raw_text[start_idx:end_idx + 1])
        except Exception as e:
            raise ValueError(f"Ошибка парсинга JSON: {str(e)}")

    def _get_fallback_metadata(self, text: str) -> dict:
        logger.warning("Активирован fallback для метаданных.")
        return {
            "category": "general_question",
            "sentiment": "neutral",
            "summary": text[:80] + "..." if len(text) > 80 else text
        }

    def process(self, text: str) -> dict:
        logger.info(f"Старт обработки запроса: {text[:50]}...")
        
        # Шаг 1: Классификация
        classifier_prompt = CLASSIFIER_USER_TEMPLATE.format(text=text)
        try:
            raw_class = self._get_completion_with_retry(CLASSIFIER_SYSTEM_PROMPT, classifier_prompt)
            parsed_class = self._clean_and_parse_json(raw_class)
            validated_meta = ClassificationSchema(**parsed_class)
            metadata = validated_meta.model_dump()
            logger.info("Метаданные успешно валидированы.")
        except Exception as e:
            logger.error(f"Сбой классификации: {e}. Применяем fallback.")
            metadata = self._get_fallback_metadata(text)

        category = metadata["category"]
        sentiment = metadata["sentiment"]
        summary = metadata["summary"]

        # Шаг 2: Генерация ответа
        system_response_prompt = ROUTING_PROMPTS.get(category, ROUTING_PROMPTS["general_question"])
        user_response_prompt = f"Напиши ответ на это обращение клиента:\n\"{text}\""

        try:
            final_answer = self._get_completion_with_retry(system_response_prompt, user_response_prompt)
        except Exception:
            logger.warning("Сбой генерации ответа. Применяем fallback.")
            final_answer = "Здравствуйте! Ваше обращение принято и передано специалисту. Скоро ответим."

        # Шаг 3: Self-Check и Рефлексия
        for attempt in range(1, 3):
            self_check_prompt = SELF_CHECK_PROMPT_TEMPLATE.format(
                original_text=text,
                final_answer=final_answer
            )
            try:
                raw_check = self._get_completion_with_retry(
                    "Ты — модератор качества поддержки.",
                    self_check_prompt
                )
                parsed_check = self._clean_and_parse_json(raw_check)
                if parsed_check.get("is_correct", True):
                    logger.info("Ответ одобрен модератором.")
                    break
                else:
                    reason = parsed_check.get("reason", "Замечание модератора.")
                    logger.warning(f"Попытка {attempt}: Ответ отклонен: {reason}")
                    
                    # Перегенерация с учетом критики
                    correction_prompt = f"""Исправь ответ поддержки на основе замечания критика: "{reason}"\n\nОригинальный запрос: "{text}" """
                    final_answer = self._get_completion_with_retry(system_response_prompt, correction_prompt)
            except Exception as e:
                logger.warning(f"Сбой на шаге модерации: {e}. Пропускаем проверку.")
                break

        # Guardrail на длину
        if len(final_answer) > 1000:
            final_answer = final_answer[:997] + "..."

        return {
            "category": category,
            "sentiment": sentiment,
            "summary": summary,
            "final_answer": final_answer
        }