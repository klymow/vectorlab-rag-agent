"""Диагностический CLI полного retrieval pipeline без LLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_settings  # noqa: E402
from app.rag.bm25_index import BM25IndexError, load_bm25_index  # noqa: E402
from app.rag.bm25_search import BM25Searcher  # noqa: E402
from app.rag.hybrid_search import HybridSearcher  # noqa: E402
from app.rag.local_embeddings import LocalEmbeddingModel  # noqa: E402
from app.rag.vector_index import VectorIndexError, load_vector_index  # noqa: E402
from app.rag.vector_search import VectorSearcher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Показать Vector, BM25 и итоговый RRF Top-K."
    )
    parser.add_argument("query", help="Русский текст или технический код.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        settings = load_settings()
        vector_artifacts = load_vector_index(settings.storage_dir)
        indexed_model = str(vector_artifacts.manifest["embedding_model"])
        if indexed_model != settings.embedding_model:
            raise VectorIndexError(
                "Модель в конфигурации не совпадает с индексом: "
                f"{settings.embedding_model!r} != {indexed_model!r}"
            )
        vector_searcher = VectorSearcher(
            vector_artifacts,
            LocalEmbeddingModel(indexed_model),
        )
        bm25_searcher = BM25Searcher(load_bm25_index(settings.storage_dir))
        response = HybridSearcher(
            vector_searcher,
            bm25_searcher,
            vector_top_n=settings.vector_top_n,
            bm25_top_n=settings.bm25_top_n,
            rrf_k=settings.rrf_k,
            final_top_k=settings.final_top_k,
        ).search(args.query)
    except (OSError, ValueError, VectorIndexError, BM25IndexError) as exc:
        print(f"ОШИБКА ПОИСКА: {exc}", file=sys.stderr)
        return 1

    print(f"Запрос: {response.query}")
    print("\n=== Vector Search ===")
    for item in response.vector_results:
        print(
            f"{item.rank}. score={item.similarity:.4f} | "
            f"{item.document} | {item.section}"
        )

    print("\n=== BM25 Search ===")
    for item in response.bm25_results:
        excluded = " | исключён из RRF: score <= 0" if item.score <= 0 else ""
        print(
            f"{item.rank}. score={item.score:.4f} | "
            f"{item.document_title} | {item.section}{excluded}"
        )

    print("\n=== Hybrid / RRF ===")
    for item in response.final_results:
        vector_rank = "—" if item.vector_rank is None else str(item.vector_rank)
        bm25_rank = "—" if item.bm25_rank is None else str(item.bm25_rank)
        print(
            f"{item.rank}. rrf={item.rrf_score:.6f} | "
            f"Vector={vector_rank} | BM25={bm25_rank} | "
            f"{item.document} | {item.section}"
        )
        print(f"   chunk_id: {item.chunk_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
