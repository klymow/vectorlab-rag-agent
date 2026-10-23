"""Проверки загрузки документов и детерминированного chunking."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.chunk_builder import write_chunks_json  # noqa: E402
from app.rag.chunker import chunk_documents  # noqa: E402
from app.rag.document_loader import (  # noqa: E402
    DocumentLoadError,
    load_document,
    load_documents,
)


DOCUMENTS_DIR = PROJECT_ROOT / "knowledge_base" / "documents"
EXPECTED_CODES = {
    "VPN-1207",
    "0x8004010F",
    "WIFI-EAP-17",
    "FS-403",
    "PRN-042",
    "AUDIO-210",
    "ORION-DB-104",
}


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = load_documents(DOCUMENTS_DIR)
        cls.chunks = chunk_documents(cls.documents)

    def test_loads_all_documents_with_unique_ids(self) -> None:
        self.assertEqual(12, len(self.documents))
        self.assertEqual(12, len({document.document_id for document in self.documents}))

    def test_extracts_existing_front_matter_metadata(self) -> None:
        document = self.documents[0]
        self.assertEqual("VL-IT-001", document.document_id)
        self.assertEqual("Удалённый доступ и VPN", document.document_title)
        self.assertEqual("1.2", document.version)
        self.assertEqual("01_vpn_access.md", document.source_file)
        self.assertIn("# Удалённый доступ и VPN", document.content)

    def test_chunks_have_required_fields_and_unique_ids(self) -> None:
        self.assertGreater(len(self.chunks), len(self.documents))
        self.assertEqual(len(self.chunks), len({chunk.chunk_id for chunk in self.chunks}))
        for chunk in self.chunks:
            for value in chunk.to_dict().values():
                self.assertTrue(value)
            expected_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            self.assertEqual(expected_hash, chunk.content_hash)
            self.assertTrue(chunk.text.startswith("## "))

    def test_repeated_build_is_deterministic(self) -> None:
        second_build = chunk_documents(load_documents(DOCUMENTS_DIR))
        self.assertEqual(
            [chunk.to_dict() for chunk in self.chunks],
            [chunk.to_dict() for chunk in second_build],
        )

    def test_error_codes_remain_in_context(self) -> None:
        corpus = "\n".join(chunk.text for chunk in self.chunks)
        for code in EXPECTED_CODES:
            self.assertIn(code, corpus)

        for code in ("VPN-1207", "0x8004010F", "FS-403", "ORION-DB-104"):
            error_chunks = [
                chunk
                for chunk in self.chunks
                if code in chunk.section and code in chunk.text
            ]
            self.assertTrue(error_chunks, f"Не найден раздел ошибки {code}")
            self.assertTrue(
                any("означает" in chunk.text and "1." in chunk.text for chunk in error_chunks),
                f"Описание или шаги отделены от кода {code}",
            )

    def test_nested_priority_subsections_are_not_lost(self) -> None:
        priorities = [
            chunk.text
            for chunk in self.chunks
            if chunk.document_id == "VL-IT-012" and chunk.section == "Приоритеты"
        ]
        combined = "\n".join(priorities)
        for heading in ("### P1 Критический", "### P2 Высокий", "### P3 Обычный", "### P4 Плановый запрос"):
            self.assertIn(heading, combined)

    def test_missing_required_metadata_reports_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "broken.md"
            path.write_text(
                "---\ndocument_id: TEST-001\ntitle: Тест\n---\n\n# Тест\n\n## Раздел\n\nТекст.",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DocumentLoadError, r"broken\.md.*version"):
                load_document(path)

    def test_json_serialization_preserves_all_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "chunks.json"
            write_chunks_json(output_path, len(self.documents), self.chunks)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(1, payload["schema_version"])
        self.assertEqual(12, payload["documents_count"])
        self.assertEqual(len(self.chunks), payload["chunks_count"])
        self.assertEqual(
            [chunk.to_dict() for chunk in self.chunks],
            payload["chunks"],
        )


if __name__ == "__main__":
    unittest.main()

