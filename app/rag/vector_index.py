"""Построение, сохранение и загрузка точного FAISS IndexFlatIP."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .local_embeddings import l2_normalize
from .models import Chunk


INDEX_FILENAME = "faiss.index"
MAPPING_FILENAME = "faiss_mapping.json"
MANIFEST_FILENAME = "vector_manifest.json"
CHUNKS_FILENAME = "chunks.json"


class VectorIndexError(ValueError):
    """Ошибка построения или согласованности векторных артефактов."""


@dataclass(frozen=True, slots=True)
class LoadedVectorIndex:
    index: Any
    chunk_ids: list[str]
    chunks_by_id: dict[str, Chunk]
    manifest: dict[str, object]


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


def create_index(vectors: np.ndarray) -> tuple[Any, np.ndarray]:
    """Создать точный IndexFlatIP из нормализованных float32-векторов."""

    normalized = l2_normalize(vectors)
    index = faiss.IndexFlatIP(int(normalized.shape[1]))
    index.add(normalized)
    return index, normalized


def save_vector_index(
    storage_dir: Path,
    chunks: list[Chunk],
    vectors: np.ndarray,
    *,
    model_name: str,
) -> dict[str, object]:
    """Сохранить индекс, позиционное соответствие и проверяемый manifest."""

    if len(chunks) != int(vectors.shape[0]):
        raise VectorIndexError(
            f"Чанков ({len(chunks)}) и embeddings ({vectors.shape[0]}) должно быть поровну"
        )
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise VectorIndexError("Нельзя построить mapping: chunk_id не уникальны")

    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = storage_dir / CHUNKS_FILENAME
    if not chunks_path.exists():
        raise VectorIndexError(
            f"Перед индексированием должен существовать файл чанков: '{chunks_path}'"
        )

    index, normalized = create_index(vectors)
    index_path = storage_dir / INDEX_FILENAME
    mapping_path = storage_dir / MAPPING_FILENAME
    manifest_path = storage_dir / MANIFEST_FILENAME

    temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
    faiss.write_index(index, str(temporary_index))
    temporary_index.replace(index_path)
    _write_json(
        mapping_path,
        {"schema_version": 1, "chunk_ids": chunk_ids},
    )

    manifest: dict[str, object] = {
        "schema_version": 1,
        "embedding_model": model_name,
        "embedding_dimension": int(normalized.shape[1]),
        "vector_count": int(normalized.shape[0]),
        "index_type": "IndexFlatIP",
        "similarity": "cosine_via_normalized_inner_product",
        "vectors_l2_normalized": True,
        "embedding_input": "document_title + section + chunk_text",
        "files": {
            "index": INDEX_FILENAME,
            "mapping": MAPPING_FILENAME,
            "chunks": CHUNKS_FILENAME,
        },
        "sha256": {
            "index": _sha256_file(index_path),
            "mapping": _sha256_file(mapping_path),
            "chunks": _sha256_file(chunks_path),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def _required_file(storage_dir: Path, filename: str) -> Path:
    path = storage_dir / filename
    if not path.exists():
        raise VectorIndexError(
            f"Не найден векторный артефакт '{path}'. Запустите scripts/reindex.py."
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
        raise VectorIndexError(
            "В chunks.json отсутствуют поля: " + ", ".join(sorted(missing))
        )
    return Chunk(**{key: data[key] for key in required})


def load_vector_index(storage_dir: Path) -> LoadedVectorIndex:
    """Загрузить артефакты и проверить их взаимное соответствие."""

    storage_dir = Path(storage_dir)
    index_path = _required_file(storage_dir, INDEX_FILENAME)
    mapping_path = _required_file(storage_dir, MAPPING_FILENAME)
    manifest_path = _required_file(storage_dir, MANIFEST_FILENAME)
    chunks_path = _required_file(storage_dir, CHUNKS_FILENAME)

    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(index_path))
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        raise VectorIndexError(f"Не удалось загрузить векторные артефакты: {exc}") from exc

    chunk_ids = mapping.get("chunk_ids")
    if not isinstance(chunk_ids, list) or not all(isinstance(item, str) for item in chunk_ids):
        raise VectorIndexError("Некорректный список chunk_ids в mapping")

    chunks = [_chunk_from_dict(item) for item in chunks_payload.get("chunks", [])]
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunks_by_id) != len(chunks):
        raise VectorIndexError("В chunks.json найдены дублирующиеся chunk_id")

    if index.ntotal != len(chunk_ids):
        raise VectorIndexError(
            f"В FAISS {index.ntotal} векторов, а в mapping {len(chunk_ids)} позиций"
        )
    if manifest.get("vector_count") != index.ntotal:
        raise VectorIndexError("vector_count в manifest не совпадает с FAISS")
    if manifest.get("embedding_dimension") != index.d:
        raise VectorIndexError("embedding_dimension в manifest не совпадает с FAISS")
    missing_chunks = [chunk_id for chunk_id in chunk_ids if chunk_id not in chunks_by_id]
    if missing_chunks:
        raise VectorIndexError(
            "Mapping ссылается на отсутствующие чанки: " + ", ".join(missing_chunks[:5])
        )

    hashes = manifest.get("sha256", {})
    for name, path in (("index", index_path), ("mapping", mapping_path), ("chunks", chunks_path)):
        if hashes.get(name) != _sha256_file(path):
            raise VectorIndexError(f"Контрольная сумма артефакта '{name}' не совпадает")

    return LoadedVectorIndex(
        index=index,
        chunk_ids=chunk_ids,
        chunks_by_id=chunks_by_id,
        manifest=manifest,
    )

