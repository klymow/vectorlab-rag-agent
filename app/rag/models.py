"""Структуры данных для документов и смысловых чанков."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    """Markdown-документ после проверки обязательных метаданных."""

    document_id: str
    document_title: str
    version: str
    source_file: str
    content: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """Смысловой фрагмент документа, готовый к будущей индексации."""

    chunk_id: str
    document_id: str
    document_title: str
    section: str
    version: str
    source_file: str
    text: str
    content_hash: str

    def to_dict(self) -> dict[str, str]:
        """Преобразовать чанк в JSON-совместимый словарь."""

        return asdict(self)

