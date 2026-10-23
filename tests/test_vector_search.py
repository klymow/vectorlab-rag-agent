"""Офлайн-проверки FAISS и формата semantic search без загрузки модели."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.chunk_builder import write_chunks_json  # noqa: E402
from app.rag.local_embeddings import (  # noqa: E402
    EmbeddingError,
    chunk_embedding_text,
    l2_normalize,
)
from app.rag.models import Chunk  # noqa: E402
from app.rag.vector_index import (  # noqa: E402
    VectorIndexError,
    load_vector_index,
    save_vector_index,
)
from app.rag.vector_search import VectorSearcher  # noqa: E402


def make_chunk(number: int) -> Chunk:
    return Chunk(
        chunk_id=f"test-{number}",
        document_id="TEST-001",
        document_title="Тестовый документ",
        section=f"Раздел {number}",
        version="1.0",
        source_file="test.md",
        text=f"## Раздел {number}\n\nТестовый текст {number}.",
        content_hash=f"{number:064x}",
    )


class FakeEmbedder:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector

    def encode_query(self, query: str) -> np.ndarray:
        return self.vector.copy()


class VectorSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temporary_dir.name)
        self.chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]
        self.vectors = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        write_chunks_json(self.storage_dir / "chunks.json", 1, self.chunks)
        save_vector_index(
            self.storage_dir,
            self.chunks,
            self.vectors,
            model_name="local-test-model",
        )

    def tearDown(self) -> None:
        self.temporary_dir.cleanup()

    def test_l2_normalization_returns_float32_unit_vectors(self) -> None:
        normalized = l2_normalize(self.vectors.astype(np.float64))
        self.assertEqual(np.float32, normalized.dtype)
        np.testing.assert_allclose(
            np.linalg.norm(normalized, axis=1),
            np.ones(3, dtype=np.float32),
            rtol=1e-6,
        )

    def test_l2_normalization_rejects_zero_vector(self) -> None:
        with self.assertRaises(EmbeddingError):
            l2_normalize(np.zeros((1, 2), dtype=np.float32))

    def test_embedding_text_contains_chunk_metadata(self) -> None:
        text = chunk_embedding_text(self.chunks[0])
        self.assertIn("Документ: Тестовый документ", text)
        self.assertIn("Раздел: Раздел 1", text)
        self.assertIn(self.chunks[0].text, text)

    def test_loads_saved_faiss_and_mapping(self) -> None:
        artifacts = load_vector_index(self.storage_dir)
        self.assertEqual(3, artifacts.index.ntotal)
        self.assertEqual(2, artifacts.index.d)
        self.assertEqual(["test-1", "test-2", "test-3"], artifacts.chunk_ids)
        self.assertEqual("IndexFlatIP", artifacts.manifest["index_type"])
        self.assertTrue(artifacts.manifest["vectors_l2_normalized"])

    def test_faiss_position_maps_to_chunk_id(self) -> None:
        artifacts = load_vector_index(self.storage_dir)
        scores, positions = artifacts.index.search(
            l2_normalize(np.asarray([[1.0, 0.0]], dtype=np.float32)),
            3,
        )
        mapped = [artifacts.chunk_ids[int(position)] for position in positions[0]]
        self.assertEqual("test-1", mapped[0])
        self.assertGreater(float(scores[0][0]), float(scores[0][-1]))

    def test_search_result_format_and_top_n_clamping(self) -> None:
        artifacts = load_vector_index(self.storage_dir)
        searcher = VectorSearcher(
            artifacts,
            FakeEmbedder(np.asarray([1.0, 0.0], dtype=np.float32)),
        )
        results = searcher.search("тестовый запрос", top_n=100)
        self.assertEqual(3, len(results))
        self.assertEqual(1, results[0].rank)
        self.assertEqual("test-1", results[0].chunk_id)
        self.assertEqual(
            {"rank", "chunk_id", "document", "section", "version", "similarity", "text"},
            set(results[0].to_dict()),
        )

    def test_missing_index_has_clear_error(self) -> None:
        empty_dir = self.storage_dir / "missing"
        empty_dir.mkdir()
        with self.assertRaisesRegex(VectorIndexError, "scripts/reindex.py"):
            load_vector_index(empty_dir)


if __name__ == "__main__":
    unittest.main()

