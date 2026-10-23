"""Семантический поиск query embedding → FAISS → chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from .local_embeddings import l2_normalize
from .vector_index import LoadedVectorIndex, VectorIndexError


class QueryEmbedder(Protocol):
    def encode_query(self, query: str) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    rank: int
    chunk_id: str
    document: str
    section: str
    version: str
    similarity: float
    text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class VectorSearcher:
    """Точный semantic search без LLM и лексического канала."""

    def __init__(self, artifacts: LoadedVectorIndex, embedder: QueryEmbedder) -> None:
        self.artifacts = artifacts
        self.embedder = embedder

    def search(self, query: str, *, top_n: int = 5) -> list[VectorSearchResult]:
        if not query.strip():
            raise ValueError("Поисковый запрос не должен быть пустым")
        if top_n <= 0:
            raise ValueError("top_n должен быть положительным")
        if self.artifacts.index.ntotal == 0:
            return []

        query_vector = np.asarray(self.embedder.encode_query(query), dtype=np.float32)
        if query_vector.ndim != 1:
            raise VectorIndexError(
                f"Embedding запроса должен быть одномерным, получена форма {query_vector.shape}"
            )
        if query_vector.shape[0] != self.artifacts.index.d:
            raise VectorIndexError(
                "Размерность запроса не совпадает с индексом: "
                f"{query_vector.shape[0]} != {self.artifacts.index.d}"
            )

        normalized_query = l2_normalize(query_vector.reshape(1, -1))
        requested = min(top_n, int(self.artifacts.index.ntotal))
        scores, positions = self.artifacts.index.search(normalized_query, requested)

        results: list[VectorSearchResult] = []
        for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
            if position < 0:
                continue
            chunk_id = self.artifacts.chunk_ids[int(position)]
            chunk = self.artifacts.chunks_by_id[chunk_id]
            results.append(
                VectorSearchResult(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    document=chunk.document_title,
                    section=chunk.section,
                    version=chunk.version,
                    similarity=float(score),
                    text=chunk.text,
                )
            )
        return results

