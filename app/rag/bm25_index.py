"""Построение и воспроизводимое хранение отдельного BM25-корпуса."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from rank_bm25 import BM25Okapi

from .models import Chunk
from .tokenizer import TOKEN_PATTERN, TOKENIZER_NAME, tokenize


CORPUS_FILENAME = "bm25_corpus.json"
MANIFEST_FILENAME = "bm25_manifest.json"
CHUNKS_FILENAME = "chunks.json"
INDEXED_FIELDS = "document_title + section + text"


class BM25IndexError(ValueError):
    """Ошибка построения или согласованности BM25-артефактов."""


@dataclass(frozen=True, slots=True)
class LoadedBM25Index:
    index: BM25Okapi
    chunk_ids: list[str]
    chunks_by_id: dict[str, Chunk]
    tokenized_corpus: list[list[str]]
    manifest: dict[str, object]


def bm25_index_text(chunk: Chunk) -> str:
    """Сформировать явно определённое индексируемое представление чанка."""

    return "\n".join((chunk.document_title, chunk.section, chunk.text))


def build_tokenized_corpus(chunks: list[Chunk]) -> list[list[str]]:
    """Токенизировать все поля корпуса тем же tokenizer, что используется для query."""

    if not chunks:
        raise BM25IndexError("Нельзя построить BM25: corpus не содержит чанков")
    corpus = [tokenize(bm25_index_text(chunk)) for chunk in chunks]
    empty = [chunks[index].chunk_id for index, tokens in enumerate(corpus) if not tokens]
    if empty:
        raise BM25IndexError(
            "После токенизации получены пустые чанки: " + ", ".join(empty[:5])
        )
    return corpus


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_bm25_index(storage_dir: Path, chunks: list[Chunk]) -> dict[str, object]:
    """Сохранить токены и mapping, достаточные для восстановления BM25Okapi."""

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise BM25IndexError("Нельзя построить BM25 mapping: chunk_id не уникальны")
    tokenized_corpus = build_tokenized_corpus(chunks)

    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = storage_dir / CHUNKS_FILENAME
    if not chunks_path.exists():
        raise BM25IndexError(
            f"Перед BM25-индексацией должен существовать файл чанков: '{chunks_path}'"
        )

    corpus_path = storage_dir / CORPUS_FILENAME
    manifest_path = storage_dir / MANIFEST_FILENAME
    corpus_payload = {
        "schema_version": 1,
        "tokenizer": {
            "name": TOKENIZER_NAME,
            "lowercase": True,
            "yo_to_e": True,
            "pattern": TOKEN_PATTERN.pattern,
        },
        "indexed_fields": INDEXED_FIELDS,
        "chunk_ids": chunk_ids,
        "tokenized_corpus": tokenized_corpus,
    }
    _write_json(corpus_path, corpus_payload)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "engine": "rank_bm25.BM25Okapi",
        "rank_bm25_version": version("rank-bm25"),
        "corpus_size": len(chunks),
        "indexed_fields": INDEXED_FIELDS,
        "tokenizer": TOKENIZER_NAME,
        "files": {
            "corpus": CORPUS_FILENAME,
            "chunks": CHUNKS_FILENAME,
        },
        "sha256": {
            "corpus": _sha256_file(corpus_path),
            "chunks": _sha256_file(chunks_path),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def _required_file(storage_dir: Path, filename: str) -> Path:
    path = storage_dir / filename
    if not path.exists():
        raise BM25IndexError(
            f"Не найден BM25-артефакт '{path}'. Запустите scripts/reindex.py."
        )
    return path


def _chunk_from_dict(data: dict[str, str]) -> Chunk:
    required = {
        "chunk_id",
        "document_id",
        "document_title",
        "section",
        "version",
        "source_file",
        "text",
        "content_hash",
    }
    missing = required.difference(data)
    if missing:
        raise BM25IndexError(
            "В chunks.json отсутствуют поля: " + ", ".join(sorted(missing))
        )
    return Chunk(**{key: data[key] for key in required})


def load_bm25_index(storage_dir: Path) -> LoadedBM25Index:
    """Восстановить BM25Okapi из сохранённого прозрачного корпуса."""

    storage_dir = Path(storage_dir)
    corpus_path = _required_file(storage_dir, CORPUS_FILENAME)
    manifest_path = _required_file(storage_dir, MANIFEST_FILENAME)
    chunks_path = _required_file(storage_dir, CHUNKS_FILENAME)
    try:
        corpus_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BM25IndexError(f"Не удалось загрузить BM25-артефакты: {exc}") from exc

    tokenizer_settings = corpus_payload.get("tokenizer", {})
    if (
        tokenizer_settings.get("name") != TOKENIZER_NAME
        or tokenizer_settings.get("pattern") != TOKEN_PATTERN.pattern
    ):
        raise BM25IndexError("Сохранённый BM25 corpus использует несовместимый tokenizer")

    chunk_ids = corpus_payload.get("chunk_ids")
    tokenized_corpus = corpus_payload.get("tokenized_corpus")
    if not isinstance(chunk_ids, list) or not all(isinstance(item, str) for item in chunk_ids):
        raise BM25IndexError("Некорректный список chunk_ids в BM25 corpus")
    if not isinstance(tokenized_corpus, list) or not all(
        isinstance(tokens, list) and all(isinstance(token, str) for token in tokens)
        for tokens in tokenized_corpus
    ):
        raise BM25IndexError("Некорректный tokenized_corpus")
    if not chunk_ids or len(chunk_ids) != len(tokenized_corpus):
        raise BM25IndexError("BM25 corpus пуст или не совпадает с mapping")

    chunks = [_chunk_from_dict(item) for item in chunks_payload.get("chunks", [])]
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunks_by_id) != len(chunks):
        raise BM25IndexError("В chunks.json найдены дублирующиеся chunk_id")
    missing_chunks = [chunk_id for chunk_id in chunk_ids if chunk_id not in chunks_by_id]
    if missing_chunks:
        raise BM25IndexError(
            "BM25 mapping ссылается на отсутствующие чанки: " + ", ".join(missing_chunks[:5])
        )
    if manifest.get("corpus_size") != len(chunk_ids):
        raise BM25IndexError("corpus_size в BM25 manifest не совпадает с corpus")

    hashes = manifest.get("sha256", {})
    for name, path in (("corpus", corpus_path), ("chunks", chunks_path)):
        if hashes.get(name) != _sha256_file(path):
            raise BM25IndexError(f"Контрольная сумма BM25-артефакта '{name}' не совпадает")

    return LoadedBM25Index(
        index=BM25Okapi(tokenized_corpus),
        chunk_ids=chunk_ids,
        chunks_by_id=chunks_by_id,
        tokenized_corpus=tokenized_corpus,
        manifest=manifest,
    )
