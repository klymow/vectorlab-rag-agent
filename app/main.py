"""FastAPI transport для локального интерфейса AI-поддержки."""

from __future__ import annotations

from threading import Lock
from typing import Protocol

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .config import PROJECT_ROOT
from .rag.factory import build_rag_service


STATIC_DIR = PROJECT_ROOT / "app" / "static"
INDEX_FILE = STATIC_DIR / "index.html"


class RAGServiceProtocol(Protocol):
    def answer(self, question: str): ...


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Вопрос не должен быть пустым")
        return normalized


class LazyServiceProvider:
    """Создать тяжёлый RAGService только при первом POST /api/ask."""

    def __init__(self, service: RAGServiceProtocol | None = None) -> None:
        self._service = service
        self._lock = Lock()

    def get(self) -> RAGServiceProtocol:
        if self._service is None:
            with self._lock:
                if self._service is None:
                    self._service = build_rag_service()
        return self._service


def get_rag_service(request: Request) -> RAGServiceProtocol:
    try:
        return request.app.state.rag_provider.get()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Сервис генерации ответа недоступен: {type(exc).__name__}",
        ) from exc


def create_app(rag_service: RAGServiceProtocol | None = None) -> FastAPI:
    application = FastAPI(
        title="AI-оператор IT-поддержки ВекторЛаб",
        version="1.0.0",
    )
    application.state.rag_provider = LazyServiceProvider(rag_service)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(INDEX_FILE, media_type="text/html; charset=utf-8")

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "vectorlab-rag"}

    @application.post("/api/ask")
    def ask(
        payload: AskRequest,
        service: RAGServiceProtocol = Depends(get_rag_service),
    ) -> dict[str, object]:
        try:
            result = service.answer(payload.question)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Не удалось сформировать ответ: {type(exc).__name__}",
            ) from exc
        response = result.to_dict()
        response["status"] = (
            "insufficient" if result.insufficient_evidence else "sufficient"
        )
        response["sufficient_evidence"] = not result.insufficient_evidence
        return response

    return application


app = create_app()
