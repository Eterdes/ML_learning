from pydantic import BaseModel, Field, field_validator
from typing import Literal

class ClassificationSchema(BaseModel):
    category: Literal["support", "feedback", "complaint", "sales", "general_question"] = Field(
        description="Категория входящего обращения"
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Эмоциональный окрас сообщения"
    )
    summary: str = Field(
        description="Краткая выжимка сути обращения (до 200 символов)"
    )

    @field_validator("summary")
    @classmethod
    def validate_summary_length(cls, v: str) -> str:
        if len(v) > 200:
            raise ValueError(f"Превышен лимит в 200 символов! Текущая длина: {len(v)}")
        return v