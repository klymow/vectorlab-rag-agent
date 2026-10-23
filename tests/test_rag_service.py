"""Офлайн-проверки контекста, evidence, Groq client и RAG service."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.bm25_search import BM25SearchResult  # noqa: E402
from app.rag.context_builder import build_source_contexts, render_context  # noqa: E402
from app.rag.evidence import assess_retrieval_evidence  # noqa: E402
from app.rag.fusion import reciprocal_rank_fusion  # noqa: E402
from app.rag.groq_client import (  # noqa: E402
    GeneratedAnswer,
    GroqClientError,
    GroqLLMClient,
    SYSTEM_PROMPT,
)
from app.rag.hybrid_search import HybridSearchResponse  # noqa: E402
from app.rag.models import Chunk  # noqa: E402
from app.rag.service import RAGService, STANDARD_NO_ANSWER  # noqa: E402
from app.rag.vector_search import VectorSearchResult  # noqa: E402


def make_chunk(number: int) -> Chunk:
    return Chunk(
        chunk_id=f"chunk-{number}",
        document_id=f"DOC-{number:03d}",
        document_title=f"Документ {number}",
        section=f"Раздел {number}",
        version="1.0",
        source_file=f"doc-{number}.md",
        text=f"Подтверждённая инструкция {number}.",
        content_hash=f"{number:064x}",
    )


def vector_result(chunk: Chunk, rank: int, score: float = 0.7) -> VectorSearchResult:
    return VectorSearchResult(
        rank=rank,
        chunk_id=chunk.chunk_id,
        document=chunk.document_title,
        section=chunk.section,
        version=chunk.version,
        similarity=score,
        text=chunk.text,
    )


def bm25_result(chunk: Chunk, rank: int, score: float = 4.0) -> BM25SearchResult:
    return BM25SearchResult(
        rank=rank,
        score=score,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_title=chunk.document_title,
        section=chunk.section,
        version=chunk.version,
        source_file=chunk.source_file,
        text=chunk.text,
    )


def make_response(chunks: list[Chunk]) -> HybridSearchResponse:
    vectors = [vector_result(chunk, rank) for rank, chunk in enumerate(chunks, 1)]
    bm25 = [bm25_result(chunks[0], 1)] if chunks else []
    fused = reciprocal_rank_fusion(vectors, bm25, final_top_k=5)
    return HybridSearchResponse(
        query="Тестовый вопрос",
        vector_results=vectors,
        bm25_results=bm25,
        fused_results=fused,
    )


class FakeHybrid:
    def __init__(self, response: HybridSearchResponse) -> None:
        self.response = response

    def search(self, question: str) -> HybridSearchResponse:
        return HybridSearchResponse(
            query=question,
            vector_results=self.response.vector_results,
            bm25_results=self.response.bm25_results,
            fused_results=self.response.fused_results,
        )


class FakeGenerator:
    def __init__(self, generated: GeneratedAnswer) -> None:
        self.generated = generated
        self.calls: list[tuple[str, str]] = []

    def generate(self, question: str, context: str) -> GeneratedAnswer:
        self.calls.append((question, context))
        return self.generated


class ContextAndServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [make_chunk(1), make_chunk(2)]
        self.registry = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.response = make_response(self.chunks)

    def test_context_has_stable_labels_and_required_metadata_without_scores(self) -> None:
        contexts = build_source_contexts(self.response.final_results, self.registry)
        self.assertEqual(["S1", "S2"], [item.label for item in contexts])
        rendered = render_context(contexts)
        self.assertIn("[S1]", rendered)
        self.assertIn("ID документа: DOC-001", rendered)
        self.assertIn("Версия: 1.0", rendered)
        self.assertNotIn("rrf", rendered.lower())
        self.assertNotIn("score", rendered.lower())

    def test_evidence_records_signals_without_claiming_a_threshold(self) -> None:
        evidence = assess_retrieval_evidence(self.response)
        self.assertEqual(1, evidence.positive_bm25_count)
        self.assertEqual(1, evidence.shared_top_n_count)
        self.assertTrue(evidence.top_hybrid_supported_by_both)
        self.assertTrue(evidence.requires_llm_validation)

    def test_only_cited_sources_are_returned(self) -> None:
        generator = FakeGenerator(
            GeneratedAnswer("sufficient", "Используйте инструкцию [S2].", ["S2"])
        )
        result = RAGService(
            FakeHybrid(self.response), generator, self.registry
        ).answer("Что делать?")
        self.assertFalse(result.insufficient_evidence)
        self.assertEqual(["S2"], [source.label for source in result.sources])
        self.assertEqual("DOC-002", result.sources[0].document_id)
        self.assertEqual(2, len(result.final_context))
        self.assertEqual(1, len(generator.calls))

    def test_llm_insufficient_is_standardized(self) -> None:
        generator = FakeGenerator(
            GeneratedAnswer("insufficient", "Не знаю.", [])
        )
        result = RAGService(
            FakeHybrid(self.response), generator, self.registry
        ).answer("Нет ли ответа?")
        self.assertTrue(result.insufficient_evidence)
        self.assertEqual(STANDARD_NO_ANSWER, result.answer)
        self.assertEqual([], result.sources)

    def test_unknown_source_label_is_rejected(self) -> None:
        generator = FakeGenerator(
            GeneratedAnswer("sufficient", "Ответ [S99].", ["S99"])
        )
        result = RAGService(
            FakeHybrid(self.response), generator, self.registry
        ).answer("Вопрос")
        self.assertTrue(result.insufficient_evidence)
        self.assertEqual("answer_references_unknown_source", result.no_answer_reason)

    def test_mismatched_or_missing_citations_are_rejected(self) -> None:
        cases = (
            GeneratedAnswer("sufficient", "Ответ без ссылки.", []),
            GeneratedAnswer("sufficient", "Ответ [S1].", ["S1", "S2"]),
        )
        for generated in cases:
            with self.subTest(generated=generated):
                result = RAGService(
                    FakeHybrid(self.response), FakeGenerator(generated), self.registry
                ).answer("Вопрос")
                self.assertTrue(result.insufficient_evidence)
                self.assertEqual(STANDARD_NO_ANSWER, result.answer)

    def test_empty_final_context_skips_llm(self) -> None:
        empty = HybridSearchResponse("Пусто", [], [], [])
        generator = FakeGenerator(
            GeneratedAnswer("sufficient", "Не должен вызываться [S1].", ["S1"])
        )
        result = RAGService(FakeHybrid(empty), generator, {}).answer("Пусто")
        self.assertTrue(result.insufficient_evidence)
        self.assertEqual([], generator.calls)
        self.assertEqual("empty_final_context", result.no_answer_reason)


class FakeGroqClient:
    def __init__(self, payload: dict[str, object], models: list[str]) -> None:
        self.payload = payload
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id=model) for model in models]
            )
        )
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self.last_request: dict[str, object] | None = None

    def _create(self, **kwargs: object) -> SimpleNamespace:
        self.last_request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(self.payload, ensure_ascii=False)
                    )
                )
            ]
        )


class GroqClientTests(unittest.TestCase):
    def test_prompt_requires_grounding_russian_and_inline_sources(self) -> None:
        self.assertIn("только на основании", SYSTEM_PROMPT)
        self.assertIn("только на русском языке", SYSTEM_PROMPT)
        self.assertIn("[S1]", SYSTEM_PROMPT)
        self.assertIn('status="sufficient"', SYSTEM_PROMPT)

    def test_missing_api_key_is_rejected_without_echoing_a_value(self) -> None:
        with self.assertRaisesRegex(GroqClientError, "GROQ_API_KEY"):
            GroqLLMClient(None, "model")

    def test_model_availability_and_structured_generation(self) -> None:
        fake = FakeGroqClient(
            {
                "status": "sufficient",
                "answer": "Ответ [S1].",
                "source_labels": ["S1"],
            },
            ["openai/gpt-oss-120b"],
        )
        client = GroqLLMClient(
            "test-key",
            "openai/gpt-oss-120b",
            client=fake,
        )
        client.check_model_available()
        generated = client.generate("Вопрос", "[S1]\nКонтекст")
        self.assertEqual("sufficient", generated.status)
        self.assertEqual(["S1"], generated.source_labels)
        self.assertEqual(
            {"type": "json_object"},
            fake.last_request["response_format"],
        )

    def test_unavailable_model_has_clear_error(self) -> None:
        client = GroqLLMClient(
            "test-key",
            "missing-model",
            client=FakeGroqClient({}, ["another-model"]),
        )
        with self.assertRaisesRegex(GroqClientError, "недоступна"):
            client.check_model_available()


if __name__ == "__main__":
    unittest.main()
