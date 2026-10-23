"""API-тесты FastAPI без загрузки embeddings и реальных Groq-вызовов."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import create_app  # noqa: E402


def result_payload(*, insufficient: bool = False) -> dict[str, object]:
    sources = [] if insufficient else [
        {
            "label": "S1",
            "document_id": "VL-IT-001",
            "document_title": "Удалённый доступ и VPN",
            "section": "Ошибка VPN-1207",
            "version": "1.2",
            "source_file": "01_vpn_access.md",
        }
    ]
    return {
        "question": "Что делать при VPN-1207?",
        "answer": (
            "В базе знаний недостаточно информации для ответа на этот вопрос."
            if insufficient
            else "Перезапустите VectorGate [S1]."
        ),
        "sources": sources,
        "vector_diagnostics": [
            {
                "rank": 1,
                "chunk_id": "chunk-1",
                "document": "Удалённый доступ и VPN",
                "section": "Ошибка VPN-1207",
                "version": "1.2",
                "similarity": 0.8,
                "text": "Текст",
            }
        ],
        "bm25_diagnostics": [
            {
                "rank": 1,
                "score": 5.0,
                "chunk_id": "chunk-1",
                "document_id": "VL-IT-001",
                "document_title": "Удалённый доступ и VPN",
                "section": "Ошибка VPN-1207",
                "version": "1.2",
                "source_file": "01_vpn_access.md",
                "text": "Текст",
            }
        ],
        "hybrid_diagnostics": [
            {
                "rank": 1,
                "chunk_id": "chunk-1",
                "document": "Удалённый доступ и VPN",
                "section": "Ошибка VPN-1207",
                "version": "1.2",
                "text": "Текст",
                "rrf_score": 0.032,
                "vector_rank": 1,
                "vector_score": 0.8,
                "bm25_rank": 1,
                "bm25_score": 5.0,
                "in_final_top_k": True,
            }
        ],
        "final_context": [
            {
                "label": "S1",
                "chunk_id": "chunk-1",
                "document_id": "VL-IT-001",
                "document_title": "Удалённый доступ и VPN",
                "section": "Ошибка VPN-1207",
                "version": "1.2",
                "source_file": "01_vpn_access.md",
                "text": "Текст",
            }
        ],
        "evidence": {
            "has_final_context": True,
            "positive_bm25_count": 1,
            "top_vector_similarity": 0.8,
            "shared_top_n_count": 1,
            "top_hybrid_supported_by_both": True,
            "requires_llm_validation": True,
            "reason": "Тест",
        },
        "insufficient_evidence": insufficient,
        "no_answer_reason": "llm_reported_insufficient_context" if insufficient else None,
    }


class FakeResult:
    def __init__(self, *, insufficient: bool = False) -> None:
        self.insufficient_evidence = insufficient
        self.payload = result_payload(insufficient=insufficient)

    def to_dict(self) -> dict[str, object]:
        return self.payload.copy()


class FakeService:
    def __init__(self, *, insufficient: bool = False, error: Exception | None = None) -> None:
        self.insufficient = insufficient
        self.error = error
        self.questions: list[str] = []

    def answer(self, question: str) -> FakeResult:
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        return FakeResult(insufficient=self.insufficient)


class APITests(unittest.TestCase):
    def test_health_does_not_call_rag_service(self) -> None:
        service = FakeService()
        with TestClient(create_app(service)) as client:
            response = client.get("/api/health")
        self.assertEqual(200, response.status_code)
        self.assertEqual({"status": "ok", "service": "vectorlab-rag"}, response.json())
        self.assertEqual([], service.questions)

    def test_root_serves_russian_ui(self) -> None:
        with TestClient(create_app(FakeService())) as client:
            response = client.get("/")
        self.assertEqual(200, response.status_code)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("AI-оператор IT-поддержки", response.text)
        self.assertIn("Диагностика поиска", response.text)

    def test_successful_ask_returns_complete_structured_response(self) -> None:
        service = FakeService()
        with TestClient(create_app(service)) as client:
            response = client.post("/api/ask", json={"question": "  Вопрос  "})
        payload = response.json()
        self.assertEqual(200, response.status_code)
        self.assertEqual(["Вопрос"], service.questions)
        self.assertEqual("sufficient", payload["status"])
        self.assertTrue(payload["sufficient_evidence"])
        for key in (
            "sources",
            "vector_diagnostics",
            "bm25_diagnostics",
            "hybrid_diagnostics",
            "final_context",
        ):
            self.assertIn(key, payload)

    def test_insufficient_is_normal_api_result_without_sources(self) -> None:
        with TestClient(create_app(FakeService(insufficient=True))) as client:
            response = client.post("/api/ask", json={"question": "CRM-999"})
        payload = response.json()
        self.assertEqual(200, response.status_code)
        self.assertEqual("insufficient", payload["status"])
        self.assertFalse(payload["sufficient_evidence"])
        self.assertEqual([], payload["sources"])

    def test_empty_question_is_rejected_before_service_call(self) -> None:
        service = FakeService()
        with TestClient(create_app(service)) as client:
            response = client.post("/api/ask", json={"question": "   "})
        self.assertEqual(422, response.status_code)
        self.assertEqual([], service.questions)

    def test_service_error_is_reported_without_internal_message(self) -> None:
        service = FakeService(error=RuntimeError("gsk_secret_should_not_escape"))
        with TestClient(create_app(service)) as client:
            response = client.post("/api/ask", json={"question": "Вопрос"})
        self.assertEqual(503, response.status_code)
        self.assertEqual(
            "Не удалось сформировать ответ: RuntimeError",
            response.json()["detail"],
        )
        self.assertNotIn("gsk_", response.text)

    def test_api_response_contains_no_secret_fields(self) -> None:
        with TestClient(create_app(FakeService())) as client:
            response = client.post("/api/ask", json={"question": "Вопрос"})
        serialized = json.dumps(response.json(), ensure_ascii=False).lower()
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("groq_api_key", serialized)
        self.assertNotIn("gsk_", serialized)

    def test_frontend_avoids_inner_html_for_model_output(self) -> None:
        script = (PROJECT_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("innerHTML", script)
        self.assertIn("answerText.textContent", script)


if __name__ == "__main__":
    unittest.main()
