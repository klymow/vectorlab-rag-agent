"""Офлайн-проверки Reciprocal Rank Fusion и Hybrid Search pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.bm25_search import BM25SearchResult  # noqa: E402
from app.rag.fusion import reciprocal_rank_fusion  # noqa: E402
from app.rag.hybrid_search import HybridSearcher  # noqa: E402
from app.rag.vector_search import VectorSearchResult  # noqa: E402


def vector_result(chunk_id: str, rank: int, score: float = 0.8) -> VectorSearchResult:
    return VectorSearchResult(
        rank=rank,
        chunk_id=chunk_id,
        document=f"Документ {chunk_id}",
        section=f"Раздел {chunk_id}",
        version="1.0",
        similarity=score,
        text=f"Текст {chunk_id}",
    )


def bm25_result(chunk_id: str, rank: int, score: float = 3.0) -> BM25SearchResult:
    return BM25SearchResult(
        rank=rank,
        score=score,
        chunk_id=chunk_id,
        document_id=f"DOC-{chunk_id}",
        document_title=f"Документ {chunk_id}",
        section=f"Раздел {chunk_id}",
        version="1.0",
        source_file=f"{chunk_id}.md",
        text=f"Текст {chunk_id}",
    )


class FakeChannel:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, top_n: int) -> list[object]:
        self.calls.append((query, top_n))
        return self.results[:top_n]


class FusionTests(unittest.TestCase):
    def test_formula_uses_rank_starting_at_one(self) -> None:
        result = reciprocal_rank_fusion([vector_result("a", 1)], [], rrf_k=60)[0]
        self.assertAlmostEqual(1.0 / 61.0, result.rrf_score)
        with self.assertRaisesRegex(ValueError, "начинаться с 1"):
            reciprocal_rank_fusion([vector_result("bad", 0)], [])

    def test_chunk_in_both_channels_gets_both_contributions(self) -> None:
        result = reciprocal_rank_fusion(
            [vector_result("same", 1)],
            [bm25_result("same", 2)],
            rrf_k=60,
        )[0]
        self.assertAlmostEqual(1.0 / 61.0 + 1.0 / 62.0, result.rrf_score)
        self.assertEqual(1, result.vector_rank)
        self.assertEqual(2, result.bm25_rank)

    def test_chunk_in_one_channel_gets_one_contribution(self) -> None:
        result = reciprocal_rank_fusion([], [bm25_result("b", 3)], rrf_k=60)[0]
        self.assertAlmostEqual(1.0 / 63.0, result.rrf_score)
        self.assertIsNone(result.vector_rank)
        self.assertEqual(3, result.bm25_rank)

    def test_deduplicates_by_chunk_id_within_and_across_channels(self) -> None:
        results = reciprocal_rank_fusion(
            [vector_result("same", 1), vector_result("same", 2)],
            [bm25_result("same", 1), bm25_result("same", 3)],
        )
        self.assertEqual(1, len(results))
        self.assertEqual(1, results[0].vector_rank)
        self.assertEqual(1, results[0].bm25_rank)

    def test_non_positive_bm25_scores_are_excluded(self) -> None:
        results = reciprocal_rank_fusion(
            [],
            [bm25_result("zero", 1, 0.0), bm25_result("negative", 2, -0.5)],
        )
        self.assertEqual([], results)

    def test_equal_scores_are_sorted_deterministically_by_chunk_id(self) -> None:
        results = reciprocal_rank_fusion(
            [vector_result("z", 1)],
            [bm25_result("a", 1)],
        )
        self.assertEqual(["a", "z"], [result.chunk_id for result in results])

    def test_final_top_k_is_marked_and_limited_by_response(self) -> None:
        vector = FakeChannel([vector_result(str(number), number) for number in range(1, 5)])
        bm25 = FakeChannel([])
        response = HybridSearcher(
            vector,
            bm25,
            vector_top_n=4,
            bm25_top_n=4,
            rrf_k=60,
            final_top_k=2,
        ).search("запрос")
        self.assertEqual(2, len(response.final_results))
        self.assertEqual([True, True, False, False], [item.in_final_top_k for item in response.fused_results])

    def test_diagnostic_ranks_and_scores_are_preserved(self) -> None:
        result = reciprocal_rank_fusion(
            [vector_result("same", 2, 0.42)],
            [bm25_result("same", 3, 7.5)],
        )[0]
        self.assertEqual(2, result.vector_rank)
        self.assertEqual(0.42, result.vector_score)
        self.assertEqual(3, result.bm25_rank)
        self.assertEqual(7.5, result.bm25_score)
        self.assertIn("in_final_top_k", result.to_dict())

    def test_pipeline_calls_both_channels_with_configured_top_n(self) -> None:
        vector = FakeChannel([vector_result("v", 1)])
        bm25 = FakeChannel([])
        searcher = HybridSearcher(
            vector,
            bm25,
            vector_top_n=8,
            bm25_top_n=7,
            rrf_k=60,
            final_top_k=5,
        )
        response = searcher.search("один канал пуст")
        self.assertEqual([("один канал пуст", 8)], vector.calls)
        self.assertEqual([("один канал пуст", 7)], bm25.calls)
        self.assertEqual(["v"], [item.chunk_id for item in response.final_results])

    def test_both_empty_channels_return_empty_final_results(self) -> None:
        searcher = HybridSearcher(
            FakeChannel([]),
            FakeChannel([]),
            vector_top_n=8,
            bm25_top_n=8,
            rrf_k=60,
            final_top_k=5,
        )
        response = searcher.search("ничего")
        self.assertEqual([], response.fused_results)
        self.assertEqual([], response.final_results)


if __name__ == "__main__":
    unittest.main()
