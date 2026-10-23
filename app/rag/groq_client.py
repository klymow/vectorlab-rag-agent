"""Единственный LLM-клиент проекта: Groq Chat Completions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from groq import Groq


SYSTEM_PROMPT = """Ты — оператор внутренней IT-поддержки.
Отвечай только на основании предоставленного контекста и только на русском языке.
Не используй внешние знания и не придумывай процедуры, коды, сроки, контакты или правила.
Не превращай предположение в подтверждённый факт.
Каждое фактическое утверждение подкрепляй ссылкой на источник вида [S1], [S2].
При status="sufficient" поле answer ОБЯЗАТЕЛЬНО должно содержать квадратные
ссылки [S1], [S2] непосредственно после подтверждаемых ими утверждений.
Поле source_labels не заменяет ссылки внутри answer.
Если контекст не подтверждает ответ на вопрос, установи status=insufficient.
Ответ должен быть практичным и кратким.
Верни только JSON-объект с полями:
status: "sufficient" или "insufficient";
answer: текст ответа;
source_labels: массив реально использованных меток без квадратных скобок.
Пример корректного формата:
{"status":"sufficient","answer":"Перезапустите клиент [S1].","source_labels":["S1"]}
"""


class GroqClientError(RuntimeError):
    """Безопасная диагностическая ошибка Groq без раскрытия API key."""


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    status: str
    answer: str
    source_labels: list[str]


class GroqLLMClient:
    """Принимает готовые question/context и не зависит от retrieval деталей."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        client: Any | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise GroqClientError("GROQ_API_KEY не настроен")
        if not model.strip():
            raise GroqClientError("GROQ_MODEL не настроен")
        self.model = model
        self._client = client if client is not None else Groq(api_key=api_key)

    def check_model_available(self) -> None:
        """Проверить, что настроенная модель видна текущему Groq key."""

        try:
            response = self._client.models.list()
            model_ids = {item.id for item in response.data}
        except Exception as exc:
            raise GroqClientError(
                f"Не удалось получить список моделей Groq: {type(exc).__name__}: {exc}"
            ) from exc
        if self.model not in model_ids:
            raise GroqClientError(
                f"Модель Groq {self.model!r} недоступна текущему API key"
            )

    def generate(self, question: str, context: str) -> GeneratedAnswer:
        """Запросить строго структурированный grounded-ответ."""

        user_prompt = f"Контекст:\n{context}\n\nВопрос пользователя:\n{question}"
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            payload = json.loads(content or "")
        except Exception as exc:
            raise GroqClientError(
                f"Ошибка генерации Groq: {type(exc).__name__}: {exc}"
            ) from exc

        status = payload.get("status")
        answer = payload.get("answer")
        labels = payload.get("source_labels")
        if status not in {"sufficient", "insufficient"}:
            raise GroqClientError("Groq вернул некорректный status")
        if not isinstance(answer, str) or not answer.strip():
            raise GroqClientError("Groq вернул пустой answer")
        if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
            raise GroqClientError("Groq вернул некорректный source_labels")
        return GeneratedAnswer(status=status, answer=answer.strip(), source_labels=labels)
