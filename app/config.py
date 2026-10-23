"""Конфигурация проекта без доступа к секретам будущих этапов."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True, slots=True)
class Settings:
    """Настройки локальных retrieval-каналов и hybrid pipeline."""

    embedding_model: str
    documents_dir: Path
    storage_dir: Path
    vector_top_n: int
    bm25_top_n: int
    rrf_k: int
    final_top_k: int
    groq_api_key: str | None = field(repr=False)
    groq_model: str


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} должен быть целым числом, получено {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} должен быть положительным, получено {value}")
    return value


def load_settings(env_path: Path | None = None) -> Settings:
    """Загрузить только настройки, нужные текущему этапу."""

    dotenv_path = env_path if env_path is not None else PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
    if not model_name:
        model_name = DEFAULT_EMBEDDING_MODEL
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip() or None
    groq_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
    if not groq_model:
        groq_model = DEFAULT_GROQ_MODEL
    return Settings(
        embedding_model=model_name,
        documents_dir=PROJECT_ROOT / "knowledge_base" / "documents",
        storage_dir=PROJECT_ROOT / "storage",
        vector_top_n=_positive_int("VECTOR_TOP_N", 8),
        bm25_top_n=_positive_int("BM25_TOP_N", 8),
        rrf_k=_positive_int("RRF_K", 60),
        final_top_k=_positive_int("FINAL_TOP_K", 5),
        groq_api_key=groq_api_key,
        groq_model=groq_model,
    )

