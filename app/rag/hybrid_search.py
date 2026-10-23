"""Pipeline query → Vector + BM25 → RRF → Final Top-K."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .bm25_search import BM25SearchResult
from .fusion import FusedSearchResult, reciprocal_rank_fusion
from .vector_search import VectorSearchResult


class VectorChannel(Protocol):
    def search(self, query: str, *, top_n: int) -> list[VectorSearchResult]: ...


class BM25Channel(Protocol):
    def search(self, query: str, *, top_n: int) -> list[BM25SearchResult]: ...


@dataclass(frozen=True, slots=True)
class HybridSearchResponse:
    query: str
    vector_results: list[VectorSearchResult]
    bm25_results: list[BM25SearchResult]
    fused_results: list[FusedSearchResult]

    @property
    def final_results(self) -> list[FusedSearchResult]:
        return [result for result in self.fused_results if result.in_final_top_k]


class HybridSearcher:
    """Запускает оба retrieval-канала и объединяет только их ранги."""

    def __init__(
        self,
        vector_searcher: VectorChannel,
        bm25_searcher: BM25Channel,
        *,
        vector_top_n: int,
        bm25_top_n: int,
        rrf_k: int,
        final_top_k: int,
    ) -> None:
        for name, value in (
            ("vector_top_n", vector_top_n),
            ("bm25_top_n", bm25_top_n),
            ("rrf_k", rrf_k),
            ("final_top_k", final_top_k),
        ):
            if value <= 0:
                raise ValueError(f"{name} должен быть положительным")
        self.vector_searcher = vector_searcher
        self.bm25_searcher = bm25_searcher
        self.vector_top_n = vector_top_n
        self.bm25_top_n = bm25_top_n
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k

    def search(self, query: str) -> HybridSearchResponse:
        if not query.strip():
            raise ValueError("Поисковый запрос не должен быть пустым")
        vector_results = self.vector_searcher.search(query, top_n=self.vector_top_n)
        bm25_results = self.bm25_searcher.search(query, top_n=self.bm25_top_n)
        fused_results = reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            rrf_k=self.rrf_k,
            final_top_k=self.final_top_k,
        )
        return HybridSearchResponse(
            query=query,
            vector_results=vector_results,
            bm25_results=bm25_results,
            fused_results=fused_results,
        )
