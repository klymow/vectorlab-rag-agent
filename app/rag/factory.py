"""Единая сборка RAGService для CLI и FastAPI."""

from __future__ import annotations

from app.config import Settings, load_settings

from .bm25_index import load_bm25_index
from .bm25_search import BM25Searcher
from .groq_client import GroqLLMClient
from .hybrid_search import HybridSearcher
from .local_embeddings import LocalEmbeddingModel
from .service import RAGService
from .vector_index import VectorIndexError, load_vector_index
from .vector_search import VectorSearcher


def build_rag_service(
    settings: Settings | None = None,
    *,
    verify_groq_model: bool = True,
) -> RAGService:
    """Собрать один service layer без дублирования pipeline в transport-коде."""

    current = settings or load_settings()
    vector_artifacts = load_vector_index(current.storage_dir)
    indexed_model = str(vector_artifacts.manifest["embedding_model"])
    if indexed_model != current.embedding_model:
        raise VectorIndexError(
            "Модель в конфигурации не совпадает с индексом: "
            f"{current.embedding_model!r} != {indexed_model!r}"
        )
    bm25_artifacts = load_bm25_index(current.storage_dir)
    hybrid = HybridSearcher(
        VectorSearcher(
            vector_artifacts,
            LocalEmbeddingModel(indexed_model),
        ),
        BM25Searcher(bm25_artifacts),
        vector_top_n=current.vector_top_n,
        bm25_top_n=current.bm25_top_n,
        rrf_k=current.rrf_k,
        final_top_k=current.final_top_k,
    )
    llm = GroqLLMClient(current.groq_api_key, current.groq_model)
    if verify_groq_model:
        llm.check_model_available()
    return RAGService(hybrid, llm, bm25_artifacts.chunks_by_id)
