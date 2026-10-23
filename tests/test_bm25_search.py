"""Офлайн-проверки tokenizer, BM25 corpus и формата результатов."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.bm25_index import (  # noqa: E402
    BM25IndexError,
    bm25_index_text,
    build_tokenized_corpus,
    load_bm25_index,
    save_bm25_index,
)
from app.rag.bm25_search import BM25Searcher  # noqa: E402
from app.rag.chunk_builder import write_chunks_json  # noqa: E402
from app.rag.models import Chunk  # noqa: E402
from app.rag.tokenizer import tokenize  # noqa: E402


def make_chunk(number: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"test-{number}",
        document_id=f"TEST-{number:03d}",
        document_title=f"Документ {number}",
        section=f"Раздел {number}",
        version="1.0",
        source_file=f"test-{number}.md",
        text=text,
        content_hash=f"{number:064x}",
    )


class TokenizerTests(unittest.TestCase):
    def test_tokenization_is_deterministic(self) -> None:
        text = "VectorGate: VPN-1207, Orion ERP."
        self.assertEqual(tokenize(text), tokenize(text))

    def test_lowercase_and_yo_normalization(self) -> None:
        self.assertEqual(["все", "еще"], tokenize("ВСЁ ЕЩЁ"))

    def test_preserves_all_control_codes(self) -> None:
        codes = (
            "VPN-1207",
            "0x8004010F",
            "WIFI-EAP-17",
            "FS-403",
            "PRN-042",
            "AUDIO-210",
            "ORION-DB-104",
            "VL-CORP",
        )
        for code in codes:
            with self.subTest(code=code):
                self.assertEqual([code.lower()], tokenize(code))

    def test_product_names_and_numbers(self) -> None:
        self.assertEqual(
            ["vectorgate", "orion", "erp", "2026"],
            tokenize("VectorGate / Orion ERP 2026"),
        )

    def test_same_tokenizer_for_corpus_and_query(self) -> None:
        chunk = make_chunk(1, "Ошибка VPN-1207 в VectorGate")
        corpus_tokens = build_tokenized_corpus([chunk])[0]
        query_tokens = tokenize("VPN-1207 VectorGate")
        self.assertTrue(set(query_tokens).issubset(corpus_tokens))
        self.assertEqual(tokenize(bm25_index_text(chunk)), corpus_tokens)


class BM25SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temporary_dir.name)
        self.chunks = [
            make_chunk(1, "Ошибка VPN-1207 означает недоступность шлюза VectorGate."),
            make_chunk(2, "Outlook показывает ошибку 0x8004010F."),
            make_chunk(3, "Принтер сообщает код PRN-042."),
        ]
        write_chunks_json(self.storage_dir / "chunks.json", 3, self.chunks)
        save_bm25_index(self.storage_dir, self.chunks)

    def tearDown(self) -> None:
        self.temporary_dir.cleanup()

    def test_result_contains_complete_chunk_metadata(self) -> None:
        artifacts = load_bm25_index(self.storage_dir)
        result = BM25Searcher(artifacts).search("VPN-1207", top_n=1)[0]
        self.assertEqual("test-1", result.chunk_id)
        self.assertEqual("TEST-001", result.document_id)
        self.assertEqual("Документ 1", result.document_title)
        self.assertEqual("Раздел 1", result.section)
        self.assertEqual("1.0", result.version)
        self.assertEqual("test-1.md", result.source_file)
        self.assertIn("VPN-1207", result.text)
        self.assertEqual(
            {
                "rank",
                "score",
                "chunk_id",
                "document_id",
                "document_title",
                "section",
                "version",
                "source_file",
                "text",
            },
            set(result.to_dict()),
        )

    def test_top_n_is_applied_and_clamped(self) -> None:
        searcher = BM25Searcher(load_bm25_index(self.storage_dir))
        self.assertEqual(2, len(searcher.search("ошибка", top_n=2)))
        self.assertEqual(3, len(searcher.search("ошибка", top_n=100)))

    def test_empty_query_is_rejected(self) -> None:
        searcher = BM25Searcher(load_bm25_index(self.storage_dir))
        for query in ("", "   ", "!!!"):
            with self.subTest(query=query):
                with self.assertRaisesRegex(ValueError, "пустым"):
                    searcher.search(query)

    def test_non_positive_top_n_is_rejected(self) -> None:
        searcher = BM25Searcher(load_bm25_index(self.storage_dir))
        with self.assertRaisesRegex(ValueError, "положительным"):
            searcher.search("VPN-1207", top_n=0)

    def test_missing_corpus_has_clear_error(self) -> None:
        missing = self.storage_dir / "missing"
        missing.mkdir()
        with self.assertRaisesRegex(BM25IndexError, "scripts/reindex.py"):
            load_bm25_index(missing)

    def test_empty_corpus_is_rejected(self) -> None:
        with self.assertRaisesRegex(BM25IndexError, "не содержит чанков"):
            build_tokenized_corpus([])

    def test_saved_corpus_is_reproducible(self) -> None:
        corpus_path = self.storage_dir / "bm25_corpus.json"
        manifest_path = self.storage_dir / "bm25_manifest.json"
        first_corpus = corpus_path.read_bytes()
        first_manifest = manifest_path.read_bytes()
        save_bm25_index(self.storage_dir, self.chunks)
        self.assertEqual(first_corpus, corpus_path.read_bytes())
        self.assertEqual(first_manifest, manifest_path.read_bytes())
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.assertEqual(
            "document_title + section + text",
            payload["indexed_fields"],
        )


if __name__ == "__main__":
    unittest.main()
