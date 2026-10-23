"""Построение и сохранение корпуса чанков."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .chunker import chunk_documents, estimate_tokens
from .document_loader import load_documents
from .models import Chunk


def build_chunks(
    documents_dir: Path,
    *,
    target_tokens: int = 600,
    max_tokens: int = 900,
) -> tuple[int, list[Chunk]]:
    """Загрузить документы и вернуть количество документов и готовые чанки."""

    documents = load_documents(documents_dir)
    chunks = chunk_documents(
        documents,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
    )
    return len(documents), chunks


def validate_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Проверить обязательные поля и детерминированные идентификаторы."""

    materialized = list(chunks)
    if not materialized:
        raise ValueError("Не сформировано ни одного чанка")

    chunk_ids: set[str] = set()
    for chunk in materialized:
        values = chunk.to_dict()
        empty = [key for key, value in values.items() if not value]
        if empty:
            raise ValueError(f"Чанк '{chunk.chunk_id}' содержит пустые поля: {', '.join(empty)}")
        if chunk.chunk_id in chunk_ids:
            raise ValueError(f"Дублирующийся chunk_id: {chunk.chunk_id}")
        if len(chunk.content_hash) != 64:
            raise ValueError(f"Некорректный SHA-256 content_hash у чанка '{chunk.chunk_id}'")
        chunk_ids.add(chunk.chunk_id)
    return materialized


def write_chunks_json(output_path: Path, document_count: int, chunks: Iterable[Chunk]) -> None:
    """Атомарно сохранить проверенные чанки в UTF-8 JSON."""

    output_path = Path(output_path)
    checked = validate_chunks(chunks)
    payload = {
        "schema_version": 1,
        "documents_count": document_count,
        "chunks_count": len(checked),
        "chunks": [chunk.to_dict() for chunk in checked],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def corpus_summary(chunks: Iterable[Chunk]) -> dict[str, object]:
    """Подготовить краткую проверочную статистику корпуса."""

    materialized = validate_chunks(chunks)
    sizes = [estimate_tokens(chunk.text) for chunk in materialized]
    by_document = Counter(chunk.document_id for chunk in materialized)
    return {
        "chunks_count": len(materialized),
        "min_estimated_tokens": min(sizes),
        "max_estimated_tokens": max(sizes),
        "average_estimated_tokens": round(sum(sizes) / len(sizes), 1),
        "chunks_by_document": dict(sorted(by_document.items())),
    }

