"""Объяснимая диагностика retrieval evidence без ненадёжного threshold."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .hybrid_search import HybridSearchResponse


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    has_final_context: bool
    positive_bm25_count: int
    top_vector_similarity: float | None
    shared_top_n_count: int
    top_hybrid_supported_by_both: bool
    requires_llm_validation: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_retrieval_evidence(response: HybridSearchResponse) -> EvidenceAssessment:
    """Собрать сигналы, не выдавая их за универсальный relevance threshold."""

    positives = [result for result in response.bm25_results if result.score > 0]
    vector_ids = {result.chunk_id for result in response.vector_results}
    shared_count = sum(result.chunk_id in vector_ids for result in positives)
    final_results = response.final_results
    top = final_results[0] if final_results else None
    top_vector_similarity = (
        response.vector_results[0].similarity if response.vector_results else None
    )
    if not final_results:
        reason = "Final Top-K пуст: подтверждающий контекст отсутствует"
        requires_llm = False
    elif not positives:
        reason = (
            "Ненулевых BM25 совпадений нет; Vector results сами по себе "
            "не доказывают наличие ответа, требуется проверка LLM по контексту"
        )
        requires_llm = True
    else:
        reason = (
            "Retrieval вернул кандидатов, но статистика контрольных запросов не даёт "
            "безопасного универсального threshold; требуется проверка LLM по контексту"
        )
        requires_llm = True
    return EvidenceAssessment(
        has_final_context=bool(final_results),
        positive_bm25_count=len(positives),
        top_vector_similarity=top_vector_similarity,
        shared_top_n_count=shared_count,
        top_hybrid_supported_by_both=bool(
            top and top.vector_rank is not None and top.bm25_rank is not None
        ),
        requires_llm_validation=requires_llm,
        reason=reason,
    )
