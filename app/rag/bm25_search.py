"""Независимый лексический поиск query → tokenizer → BM25 → chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .bm25_index import LoadedBM25Index
from .tokenizer import tokenize


@dataclass(frozen=True, slots=True)
class BM25SearchResult:
    rank: int
    score: float
    chunk_id: str
    document_id: str
    document_title: str
    section: str
    version: str
    source_file: str
    text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BM25Searcher:
    """Лексический поиск без обращения к embeddings, FAISS или LLM."""

    def __init__(self, artifacts: LoadedBM25Index) -> None:
        self.artifacts = artifacts

    def search(self, query: str, *, top_n: int = 5) -> list[BM25SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("Поисковый запрос не должен быть пустым")
        if top_n <= 0:
            raise ValueError("top_n должен быть положительным")

        scores = self.artifacts.index.get_scores(query_tokens)
        requested = min(top_n, len(self.artifacts.chunk_ids))
        positions = sorted(
            range(len(self.artifacts.chunk_ids)),
            key=lambda position: (-float(scores[position]), position),
        )[:requested]

        results: list[BM25SearchResult] = []
        for rank, position in enumerate(positions, start=1):
            chunk_id = self.artifacts.chunk_ids[position]
            chunk = self.artifacts.chunks_by_id[chunk_id]
            results.append(
                BM25SearchResult(
                    rank=rank,
                    score=float(scores[position]),
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    section=chunk.section,
                    version=chunk.version,
                    source_file=chunk.source_file,
                    text=chunk.text,
                )
            )
        return results
