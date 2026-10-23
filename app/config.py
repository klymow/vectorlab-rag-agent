"""Конфигурация проекта без доступа к секретам будущих этапов."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


@dataclass(frozen=True, slots=True)
class Settings:
    """Настройки локального контура embeddings и хранения индекса."""

    embedding_model: str
    documents_dir: Path
    storage_dir: Path


def load_settings(env_path: Path | None = None) -> Settings:
    """Загрузить только настройки, нужные текущему этапу."""

    dotenv_path = env_path if env_path is not None else PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
    if not model_name:
        model_name = DEFAULT_EMBEDDING_MODEL
    return Settings(
        embedding_model=model_name,
        documents_dir=PROJECT_ROOT / "knowledge_base" / "documents",
        storage_dir=PROJECT_ROOT / "storage",
    )

