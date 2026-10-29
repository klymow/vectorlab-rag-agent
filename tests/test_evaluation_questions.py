"""Проверка целостности набора вопросов для ручного end-to-end QA."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "tests" / "evaluation_questions.json"
DOCUMENTS_DIR = PROJECT_ROOT / "knowledge_base" / "documents"


def test_evaluation_questions_cover_every_document_and_no_answer() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    assert len(questions) >= 15
    assert len({item["question"] for item in questions}) == len(questions)
    assert {item["expected_status"] for item in questions} == {
        "sufficient",
        "insufficient",
    }

    expected_documents = {
        line.split(":", 1)[1].strip()
        for path in DOCUMENTS_DIR.glob("*.md")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("document_id:")
    }
    covered_documents = {
        document_id
        for item in questions
        if item["expected_status"] == "sufficient"
        for document_id in item["expected_document_ids"]
    }
    assert covered_documents == expected_documents
    assert all(
        not item["expected_document_ids"]
        for item in questions
        if item["expected_status"] == "insufficient"
    )
