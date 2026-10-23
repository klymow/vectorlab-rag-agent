"""Сформировать storage/chunks.json из Markdown-базы знаний."""

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

from app.rag.chunk_builder import (  # noqa: E402
    build_chunks,
    corpus_summary,
    write_chunks_json,
)
from app.rag.chunker import ChunkingError  # noqa: E402
from app.rag.document_loader import DocumentLoadError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Построить смысловые чанки из Markdown-документов ВекторЛаб."
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=PROJECT_ROOT / "knowledge_base" / "documents",
        help="Каталог исходных Markdown-документов.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "storage" / "chunks.json",
        help="Путь к формируемому JSON-файлу.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=600,
        help="Ориентир размера при делении слишком крупных разделов.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help="Порог, после которого раздел разрешено дополнительно делить.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Проверить построение чанков без записи JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        document_count, chunks = build_chunks(
            args.documents_dir,
            target_tokens=args.target_tokens,
            max_tokens=args.max_tokens,
        )
        summary = corpus_summary(chunks)
        if not args.check_only:
            write_chunks_json(args.output, document_count, chunks)
    except (DocumentLoadError, ChunkingError, OSError, ValueError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1

    print(f"Документов прочитано: {document_count}")
    print(f"Чанков сформировано: {summary['chunks_count']}")
    print(
        "Оценка размера чанков: "
        f"min={summary['min_estimated_tokens']}, "
        f"avg={summary['average_estimated_tokens']}, "
        f"max={summary['max_estimated_tokens']} токенов"
    )
    for document_id, count in summary["chunks_by_document"].items():
        print(f"  {document_id}: {count}")
    if args.check_only:
        print("Режим проверки: файл не записывался.")
    else:
        print(f"Результат сохранён: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

