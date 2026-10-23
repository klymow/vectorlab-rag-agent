"""Формирование цитируемого Final Top-K контекста без retrieval scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .fusion import FusedSearchResult
from .models import Chunk


@dataclass(frozen=True, slots=True)
class SourceContext:
    label: str
    chunk_id: str
    document_id: str
    document_title: str
    section: str
    version: str
    source_file: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_source_contexts(
    final_results: list[FusedSearchResult],
    chunks_by_id: dict[str, Chunk],
) -> list[SourceContext]:
    """Присвоить стабильные S1... в порядке итогового RRF rank."""

    contexts: list[SourceContext] = []
    for index, result in enumerate(final_results, start=1):
        try:
            chunk = chunks_by_id[result.chunk_id]
        except KeyError as exc:
            raise ValueError(
                f"Final Top-K ссылается на отсутствующий chunk_id: {result.chunk_id}"
            ) from exc
        contexts.append(
            SourceContext(
                label=f"S{index}",
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                section=chunk.section,
                version=chunk.version,
                source_file=chunk.source_file,
                text=chunk.text,
            )
        )
    return contexts


def render_context(contexts: list[SourceContext]) -> str:
    """Представить контекст для LLM без диагностических score и rank."""

    blocks = []
    for source in contexts:
        blocks.append(
            "\n".join(
                (
                    f"[{source.label}]",
                    f"Документ: {source.document_title}",
                    f"ID документа: {source.document_id}",
                    f"Раздел: {source.section}",
                    f"Версия: {source.version}",
                    "Текст:",
                    source.text,
                )
            )
        )
    return "\n\n".join(blocks)
