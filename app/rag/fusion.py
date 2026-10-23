"""Прозрачное объединение Vector и BM25 результатов через RRF."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .bm25_search import BM25SearchResult
from .vector_search import VectorSearchResult


@dataclass(frozen=True, slots=True)
class FusedSearchResult:
    rank: int
    chunk_id: str
    document: str
    section: str
    version: str
    text: str
    rrf_score: float
    vector_rank: int | None
    vector_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    in_final_top_k: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_ranks(results: Iterable[object], channel: str) -> None:
    for result in results:
        rank = getattr(result, "rank", None)
        if not isinstance(rank, int) or rank < 1:
            raise ValueError(f"Rank канала {channel} должен начинаться с 1")


def reciprocal_rank_fusion(
    vector_results: list[VectorSearchResult],
    bm25_results: list[BM25SearchResult],
    *,
    rrf_k: int = 60,
    final_top_k: int = 5,
) -> list[FusedSearchResult]:
    """Вернуть все уникальные candidates, пометив элементы итогового Top-K."""

    if rrf_k <= 0:
        raise ValueError("rrf_k должен быть положительным")
    if final_top_k <= 0:
        raise ValueError("final_top_k должен быть положительным")
    _validate_ranks(vector_results, "Vector")
    _validate_ranks(bm25_results, "BM25")

    candidates: dict[str, dict[str, object]] = {}
    seen_vector: set[str] = set()
    for result in sorted(vector_results, key=lambda item: (item.rank, item.chunk_id)):
        if result.chunk_id in seen_vector:
            continue
        seen_vector.add(result.chunk_id)
        candidates[result.chunk_id] = {
            "chunk_id": result.chunk_id,
            "document": result.document,
            "section": result.section,
            "version": result.version,
            "text": result.text,
            "rrf_score": 1.0 / (rrf_k + result.rank),
            "vector_rank": result.rank,
            "vector_score": result.similarity,
            "bm25_rank": None,
            "bm25_score": None,
        }

    seen_bm25: set[str] = set()
    for result in sorted(bm25_results, key=lambda item: (item.rank, item.chunk_id)):
        if result.score <= 0 or result.chunk_id in seen_bm25:
            continue
        seen_bm25.add(result.chunk_id)
        candidate = candidates.get(result.chunk_id)
        contribution = 1.0 / (rrf_k + result.rank)
        if candidate is None:
            candidates[result.chunk_id] = {
                "chunk_id": result.chunk_id,
                "document": result.document_title,
                "section": result.section,
                "version": result.version,
                "text": result.text,
                "rrf_score": contribution,
                "vector_rank": None,
                "vector_score": None,
                "bm25_rank": result.rank,
                "bm25_score": result.score,
            }
        else:
            candidate["rrf_score"] = float(candidate["rrf_score"]) + contribution
            candidate["bm25_rank"] = result.rank
            candidate["bm25_score"] = result.score

    ordered = sorted(
        candidates.values(),
        key=lambda item: (-float(item["rrf_score"]), str(item["chunk_id"])),
    )
    return [
        FusedSearchResult(
            rank=rank,
            in_final_top_k=rank <= final_top_k,
            **candidate,
        )
        for rank, candidate in enumerate(ordered, start=1)
    ]
