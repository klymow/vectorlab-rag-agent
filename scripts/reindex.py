"""Полностью перестроить чанки, локальные embeddings и FAISS-индекс."""

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
from app.rag.chunk_builder import build_chunks, write_chunks_json  # noqa: E402
from app.rag.local_embeddings import LocalEmbeddingModel, chunk_embedding_text  # noqa: E402
from app.rag.vector_index import save_vector_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Перестроить локальный semantic-search индекс ВекторЛаб."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", help="Переопределить локальную embedding-модель.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    model_name = args.model or settings.embedding_model
    try:
        document_count, chunks = build_chunks(settings.documents_dir)
        chunks_path = settings.storage_dir / "chunks.json"
        write_chunks_json(chunks_path, document_count, chunks)

        embedder = LocalEmbeddingModel(model_name, batch_size=args.batch_size)
        vectors = embedder.encode(
            [chunk_embedding_text(chunk) for chunk in chunks],
            show_progress=True,
        )
        manifest = save_vector_index(
            settings.storage_dir,
            chunks,
            vectors,
            model_name=model_name,
        )
    except Exception as exc:
        print(f"ОШИБКА ИНДЕКСАЦИИ: {exc}", file=sys.stderr)
        return 1

    print(f"Документов обработано: {document_count}")
    print(f"Чанков проиндексировано: {len(chunks)}")
    print(f"Embedding-модель: {manifest['embedding_model']}")
    print(f"Размерность embeddings: {manifest['embedding_dimension']}")
    print(f"FAISS-индекс: {manifest['index_type']}")
    for filename in ("chunks.json", "faiss.index", "faiss_mapping.json", "vector_manifest.json"):
        path = settings.storage_dir / filename
        print(f"  {filename}: {path.stat().st_size} байт")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

