"""Смысловой chunking Markdown по структуре заголовков."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from .models import Chunk, KnowledgeDocument


H2_PATTERN = re.compile(r"^##\s+(.+?)\s*$")
H3_PATTERN = re.compile(r"^###\s+(.+?)\s*$")
LIST_PATTERN = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)")
TOKEN_PATTERN = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯA-ZЁ0-9`«])")


class ChunkingError(ValueError):
    """Ошибка структуры Markdown или построения чанков."""


@dataclass(frozen=True, slots=True)
class _Section:
    title: str
    text: str
    occurrence: int


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    normalized = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", normalized)


def estimate_tokens(text: str) -> int:
    """Детерминированная оценка числа токенов без внешнего токенизатора."""

    return len(TOKEN_PATTERN.findall(text))


def _extract_sections(document: KnowledgeDocument) -> list[_Section]:
    lines = document.content.splitlines()
    sections: list[_Section] = []
    preamble: list[str] = []
    current_title: str | None = None
    current_lines: list[str] = []
    occurrences: dict[str, int] = {}

    def append_section(title: str, body_lines: list[str]) -> None:
        normalized_title = " ".join(title.split())
        occurrence = occurrences.get(normalized_title.casefold(), 0) + 1
        occurrences[normalized_title.casefold()] = occurrence
        text = _normalize_text("\n".join(body_lines))
        if not text:
            raise ChunkingError(
                f"Документ '{document.source_file}': раздел '{normalized_title}' не содержит текста"
            )
        sections.append(_Section(normalized_title, text, occurrence))

    for line in lines:
        match = H2_PATTERN.match(line)
        if match:
            if current_title is not None:
                append_section(current_title, current_lines)
            current_title = match.group(1)
            current_lines = [line]
        elif current_title is None:
            if not line.startswith("# ") and line.strip():
                preamble.append(line)
        else:
            current_lines.append(line)

    if current_title is not None:
        append_section(current_title, current_lines)

    if preamble:
        preamble_text = _normalize_text("## Введение\n\n" + "\n".join(preamble))
        sections.insert(0, _Section("Введение", preamble_text, 1))

    if not sections:
        raise ChunkingError(
            f"Документ '{document.source_file}': не найдено ни одного смыслового раздела '## ...'"
        )
    return sections


def _split_into_blocks(text: str) -> list[str]:
    raw_blocks = [_normalize_text(block) for block in re.split(r"\n\s*\n", text) if block.strip()]
    merged: list[str] = []
    for block in raw_blocks:
        starts_with_list = bool(LIST_PATTERN.match(block))
        if starts_with_list and merged and merged[-1].rstrip().endswith(":"):
            merged[-1] = f"{merged[-1]}\n\n{block}"
        else:
            merged.append(block)
    return merged


def _split_long_paragraph(block: str, max_tokens: int) -> list[str]:
    if estimate_tokens(block) <= max_tokens or LIST_PATTERN.match(block):
        return [block]

    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY.split(block) if sentence.strip()]
    if len(sentences) <= 1:
        return [block]

    parts: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*current, sentence])
        if current and estimate_tokens(candidate) > max_tokens:
            parts.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        parts.append(" ".join(current))
    return parts


def _pack_blocks(
    heading: str,
    blocks: Iterable[str],
    *,
    target_tokens: int,
    max_tokens: int,
) -> list[str]:
    expanded: list[str] = []
    for block in blocks:
        expanded.extend(_split_long_paragraph(block, max_tokens))

    parts: list[str] = []
    current: list[str] = []
    for block in expanded:
        candidate_body = "\n\n".join([*current, block])
        candidate = f"{heading}\n\n{candidate_body}" if candidate_body else heading
        if current and estimate_tokens(candidate) > max_tokens:
            parts.append(_normalize_text(f"{heading}\n\n" + "\n\n".join(current)))
            current = [block]
        else:
            current.append(block)

        current_text = _normalize_text(f"{heading}\n\n" + "\n\n".join(current))
        if estimate_tokens(current_text) >= target_tokens:
            parts.append(current_text)
            current = []

    if current:
        parts.append(_normalize_text(f"{heading}\n\n" + "\n\n".join(current)))
    return parts


def _split_by_h3(section: _Section) -> tuple[str, list[str]]:
    lines = section.text.splitlines()
    parent_heading = lines[0]
    groups: list[list[str]] = []
    current: list[str] = []

    for line in lines[1:]:
        if H3_PATTERN.match(line):
            if any(item.strip() for item in current):
                groups.append(current)
            current = [line]
        else:
            current.append(line)
    if any(item.strip() for item in current):
        groups.append(current)

    return parent_heading, [_normalize_text("\n".join(group)) for group in groups]


def _split_section(section: _Section, *, target_tokens: int, max_tokens: int) -> list[str]:
    if estimate_tokens(section.text) <= max_tokens:
        return [section.text]

    parent_heading, h3_groups = _split_by_h3(section)
    has_h3 = any(H3_PATTERN.match(group.splitlines()[0]) for group in h3_groups if group)
    if has_h3:
        return _pack_blocks(
            parent_heading,
            h3_groups,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        )

    body = "\n".join(section.text.splitlines()[1:]).strip()
    blocks = _split_into_blocks(body)
    return _pack_blocks(
        parent_heading,
        blocks,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
    )


def _section_key(document_id: str, section: _Section) -> str:
    identity = f"{document_id.casefold()}|{section.title.casefold()}|{section.occurrence}"
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]


def chunk_document(
    document: KnowledgeDocument,
    *,
    target_tokens: int = 600,
    max_tokens: int = 900,
) -> list[Chunk]:
    """Разделить документ по H2 и только при необходимости дробить крупные разделы."""

    if target_tokens <= 0 or max_tokens <= 0 or target_tokens > max_tokens:
        raise ValueError("Ожидается 0 < target_tokens <= max_tokens")

    chunks: list[Chunk] = []
    for section in _extract_sections(document):
        parts = _split_section(
            section,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        )
        section_key = _section_key(document.document_id, section)
        for part_number, part in enumerate(parts, start=1):
            text = _normalize_text(part)
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunk_id = (
                f"{document.document_id.casefold()}--{section_key}--{part_number:02d}"
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    document_title=document.document_title,
                    section=section.title,
                    version=document.version,
                    source_file=document.source_file,
                    text=text,
                    content_hash=content_hash,
                )
            )
    return chunks


def chunk_documents(
    documents: Iterable[KnowledgeDocument],
    *,
    target_tokens: int = 600,
    max_tokens: int = 900,
) -> list[Chunk]:
    """Построить чанки для корпуса и проверить уникальность идентификаторов."""

    chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    for document in documents:
        for chunk in chunk_document(
            document,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        ):
            if chunk.chunk_id in seen_ids:
                raise ChunkingError(f"Получен дублирующийся chunk_id: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)
    return chunks

