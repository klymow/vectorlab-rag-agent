"""Диагностический CLI RAG pipeline с генерацией ответа через Groq."""

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
from app.rag.groq_client import GroqClientError, GroqLLMClient  # noqa: E402
from app.rag.hybrid_search import HybridSearcher  # noqa: E402
from app.rag.local_embeddings import LocalEmbeddingModel  # noqa: E402
from app.rag.service import RAGService  # noqa: E402
from app.rag.vector_index import VectorIndexError, load_vector_index  # noqa: E402
from app.rag.vector_search import VectorSearcher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Получить grounded-ответ из локального Hybrid Search и Groq."
    )
    parser.add_argument("question", help="Вопрос внутренней IT-поддержке.")
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
        bm25_artifacts = load_bm25_index(settings.storage_dir)
        hybrid = HybridSearcher(
            VectorSearcher(
                vector_artifacts,
                LocalEmbeddingModel(indexed_model),
            ),
            BM25Searcher(bm25_artifacts),
            vector_top_n=settings.vector_top_n,
            bm25_top_n=settings.bm25_top_n,
            rrf_k=settings.rrf_k,
            final_top_k=settings.final_top_k,
        )
        llm = GroqLLMClient(settings.groq_api_key, settings.groq_model)
        llm.check_model_available()
        result = RAGService(
            hybrid,
            llm,
            bm25_artifacts.chunks_by_id,
        ).answer(args.question)
    except (OSError, ValueError, VectorIndexError, BM25IndexError, GroqClientError) as exc:
        print(f"ОШИБКА RAG: {exc}", file=sys.stderr)
        return 1

    print(f"Вопрос: {result.question}")
    print(f"Evidence: {'insufficient' if result.insufficient_evidence else 'sufficient'}")
    print(f"Причина: {result.evidence.reason}")
    print("\nОтвет:")
    print(result.answer)
    print("\nИсточники:")
    if not result.sources:
        print("  —")
    for source in result.sources:
        print(
            f"  [{source.label}] {source.document_id} | "
            f"{source.document_title} | {source.section} | "
            f"v{source.version} | {source.source_file}"
        )
    print("\nRetrieval:")
    print(
        f"  Vector={len(result.vector_diagnostics)}, "
        f"BM25={len(result.bm25_diagnostics)}, "
        f"RRF candidates={len(result.hybrid_diagnostics)}, "
        f"Final context={len(result.final_context)}"
    )
    print(
        f"  positive BM25={result.evidence.positive_bm25_count}, "
        f"shared Top-N={result.evidence.shared_top_n_count}, "
        f"top vector={result.evidence.top_vector_similarity}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
