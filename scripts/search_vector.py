"""Диагностический CLI локального семантического поиска."""

from __future__ import annotations

import argparse
import re
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
from app.rag.local_embeddings import LocalEmbeddingModel  # noqa: E402
from app.rag.vector_index import VectorIndexError, load_vector_index  # noqa: E402
from app.rag.vector_search import VectorSearcher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Найти релевантные чанки через FAISS.")
    parser.add_argument("query", help="Русский текст вопроса.")
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def _excerpt(text: str, limit: int = 260) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def main() -> int:
    args = parse_args()
    settings = load_settings()
    try:
        artifacts = load_vector_index(settings.storage_dir)
        indexed_model = str(artifacts.manifest["embedding_model"])
        if indexed_model != settings.embedding_model:
            raise VectorIndexError(
                "Модель в конфигурации не совпадает с индексом: "
                f"{settings.embedding_model!r} != {indexed_model!r}"
            )
        embedder = LocalEmbeddingModel(indexed_model)
        results = VectorSearcher(artifacts, embedder).search(
            args.query,
            top_n=args.top_n,
        )
    except (OSError, ValueError, VectorIndexError) as exc:
        print(f"ОШИБКА ПОИСКА: {exc}", file=sys.stderr)
        return 1

    print(f"Запрос: {args.query}")
    print(f"Модель: {indexed_model}")
    for result in results:
        print(
            f"\n{result.rank}. similarity={result.similarity:.4f} | "
            f"{result.document} | {result.section} | v{result.version}"
        )
        print(f"   chunk_id: {result.chunk_id}")
        print(f"   {_excerpt(result.text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

