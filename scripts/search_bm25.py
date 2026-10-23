"""Диагностический CLI независимого BM25-поиска."""

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
from app.rag.bm25_index import BM25IndexError, load_bm25_index  # noqa: E402
from app.rag.bm25_search import BM25Searcher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Найти релевантные чанки через BM25.")
    parser.add_argument("query", help="Русский текст или технический код.")
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def _excerpt(text: str, limit: int = 260) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def main() -> int:
    args = parse_args()
    settings = load_settings()
    try:
        artifacts = load_bm25_index(settings.storage_dir)
        results = BM25Searcher(artifacts).search(args.query, top_n=args.top_n)
    except (OSError, ValueError, BM25IndexError) as exc:
        print(f"ОШИБКА ПОИСКА: {exc}", file=sys.stderr)
        return 1

    print(f"Запрос: {args.query}")
    print(f"BM25 corpus: {artifacts.manifest['corpus_size']} чанков")
    for result in results:
        print(
            f"\n{result.rank}. score={result.score:.4f} | "
            f"{result.document_id} | {result.document_title} | "
            f"{result.section} | v{result.version}"
        )
        print(f"   source_file: {result.source_file}")
        print(f"   chunk_id: {result.chunk_id}")
        print(f"   {_excerpt(result.text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
