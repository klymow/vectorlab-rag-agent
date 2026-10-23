"""Верхнеуровневый RAG service: retrieval → context → Groq → sources."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Protocol

from .context_builder import SourceContext, build_source_contexts, render_context
from .evidence import EvidenceAssessment, assess_retrieval_evidence
from .groq_client import GeneratedAnswer
from .hybrid_search import HybridSearchResponse, HybridSearcher
from .models import Chunk


STANDARD_NO_ANSWER = "В базе знаний недостаточно информации для ответа на этот вопрос."
SOURCE_REFERENCE_PATTERN = re.compile(r"\[(S[1-9][0-9]*)\]")


class AnswerGenerator(Protocol):
    def generate(self, question: str, context: str) -> GeneratedAnswer: ...


@dataclass(frozen=True, slots=True)
class UsedSource:
    label: str
    document_id: str
    document_title: str
    section: str
    version: str
    source_file: str


@dataclass(frozen=True, slots=True)
class RAGAnswerResult:
    question: str
    answer: str
    sources: list[UsedSource]
    vector_diagnostics: list[dict[str, object]]
    bm25_diagnostics: list[dict[str, object]]
    hybrid_diagnostics: list[dict[str, object]]
    final_context: list[dict[str, str]]
    evidence: EvidenceAssessment
    insufficient_evidence: bool
    no_answer_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


class RAGService:
    """Оркестрирует retrieval и генерацию, не смешивая их внутренние детали."""

    def __init__(
        self,
        hybrid_searcher: HybridSearcher,
        generator: AnswerGenerator,
        chunks_by_id: dict[str, Chunk],
    ) -> None:
        self.hybrid_searcher = hybrid_searcher
        self.generator = generator
        self.chunks_by_id = chunks_by_id

    def _result(
        self,
        response: HybridSearchResponse,
        contexts: list[SourceContext],
        evidence: EvidenceAssessment,
        *,
        answer: str,
        sources: list[UsedSource],
        insufficient: bool,
        reason: str | None,
    ) -> RAGAnswerResult:
        return RAGAnswerResult(
            question=response.query,
            answer=answer,
            sources=sources,
            vector_diagnostics=[item.to_dict() for item in response.vector_results],
            bm25_diagnostics=[item.to_dict() for item in response.bm25_results],
            hybrid_diagnostics=[item.to_dict() for item in response.fused_results],
            final_context=[item.to_dict() for item in contexts],
            evidence=evidence,
            insufficient_evidence=insufficient,
            no_answer_reason=reason,
        )

    def _no_answer(
        self,
        response: HybridSearchResponse,
        contexts: list[SourceContext],
        evidence: EvidenceAssessment,
        reason: str,
    ) -> RAGAnswerResult:
        return self._result(
            response,
            contexts,
            evidence,
            answer=STANDARD_NO_ANSWER,
            sources=[],
            insufficient=True,
            reason=reason,
        )

    def answer(self, question: str) -> RAGAnswerResult:
        response = self.hybrid_searcher.search(question)
        evidence = assess_retrieval_evidence(response)
        contexts = build_source_contexts(response.final_results, self.chunks_by_id)
        if not contexts:
            return self._no_answer(
                response,
                contexts,
                evidence,
                "empty_final_context",
            )

        generated = self.generator.generate(question, render_context(contexts))
        if generated.status == "insufficient":
            return self._no_answer(
                response,
                contexts,
                evidence,
                "llm_reported_insufficient_context",
            )

        cited_labels = _ordered_unique(SOURCE_REFERENCE_PATTERN.findall(generated.answer))
        declared_labels = _ordered_unique(generated.source_labels)
        known = {source.label: source for source in contexts}
        if not cited_labels:
            return self._no_answer(
                response,
                contexts,
                evidence,
                "answer_has_no_source_references",
            )
        if any(label not in known for label in cited_labels + declared_labels):
            return self._no_answer(
                response,
                contexts,
                evidence,
                "answer_references_unknown_source",
            )
        if set(cited_labels) != set(declared_labels):
            return self._no_answer(
                response,
                contexts,
                evidence,
                "source_references_do_not_match_source_labels",
            )

        sources = [
            UsedSource(
                label=label,
                document_id=known[label].document_id,
                document_title=known[label].document_title,
                section=known[label].section,
                version=known[label].version,
                source_file=known[label].source_file,
            )
            for label in cited_labels
        ]
        return self._result(
            response,
            contexts,
            evidence,
            answer=generated.answer,
            sources=sources,
            insufficient=False,
            reason=None,
        )
