"""Загрузка Markdown-документов базы знаний ВекторЛаб."""

from __future__ import annotations

import re
from pathlib import Path

from .models import KnowledgeDocument


REQUIRED_METADATA = ("document_id", "title", "version")
H1_PATTERN = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


class DocumentLoadError(ValueError):
    """Ошибка формата или метаданных исходного документа."""


def _error(path: Path, message: str) -> DocumentLoadError:
    return DocumentLoadError(f"Ошибка документа '{path}': {message}")


def _parse_front_matter(text: str, path: Path) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise _error(path, "ожидался блок метаданных, начинающийся строкой '---'")

    closing_index: int | None = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break

    if closing_index is None:
        raise _error(path, "не найдена завершающая строка '---' блока метаданных")

    metadata: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines[1:closing_index], start=2):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise _error(path, f"некорректная строка метаданных {line_number}: {raw_line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise _error(path, f"пустой ключ или значение в строке метаданных {line_number}")
        if key in metadata:
            raise _error(path, f"метаданные '{key}' указаны более одного раза")
        metadata[key] = value

    missing = [key for key in REQUIRED_METADATA if not metadata.get(key)]
    if missing:
        raise _error(path, f"отсутствуют обязательные метаданные: {', '.join(missing)}")

    body = "\n".join(lines[closing_index + 1 :]).strip()
    if not body:
        raise _error(path, "после метаданных отсутствует содержимое")

    heading = H1_PATTERN.search(body)
    if heading is None:
        raise _error(path, "не найден заголовок документа первого уровня '# ...'")
    if heading.group(1).strip() != metadata["title"]:
        raise _error(
            path,
            "заголовок H1 не совпадает с полем title: "
            f"{heading.group(1).strip()!r} != {metadata['title']!r}",
        )

    return metadata, body


def load_document(path: Path) -> KnowledgeDocument:
    """Прочитать и проверить один Markdown-документ."""

    path = Path(path)
    if path.suffix.lower() != ".md":
        raise _error(path, "на этом этапе поддерживаются только файлы .md")

    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise _error(path, f"не удалось прочитать файл: {exc}") from exc

    metadata, body = _parse_front_matter(text, path)
    return KnowledgeDocument(
        document_id=metadata["document_id"],
        document_title=metadata["title"],
        version=str(metadata["version"]),
        source_file=path.name,
        content=body,
    )


def load_documents(documents_dir: Path) -> list[KnowledgeDocument]:
    """Найти и загрузить все Markdown-документы в стабильном порядке."""

    documents_dir = Path(documents_dir)
    if not documents_dir.exists():
        raise DocumentLoadError(f"Каталог базы знаний не найден: '{documents_dir}'")
    if not documents_dir.is_dir():
        raise DocumentLoadError(f"Путь базы знаний не является каталогом: '{documents_dir}'")

    paths = sorted(documents_dir.glob("*.md"), key=lambda item: item.name.casefold())
    if not paths:
        raise DocumentLoadError(f"В каталоге '{documents_dir}' нет Markdown-документов")

    documents = [load_document(path) for path in paths]
    seen: dict[str, str] = {}
    for document in documents:
        previous = seen.get(document.document_id)
        if previous is not None:
            raise DocumentLoadError(
                f"Дублирующийся document_id '{document.document_id}' "
                f"в файлах '{previous}' и '{document.source_file}'"
            )
        seen[document.document_id] = document.source_file

    return documents

