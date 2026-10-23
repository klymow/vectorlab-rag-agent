"""Прозрачная токенизация русского и технического текста для BM25."""

from __future__ import annotations

import re


TOKENIZER_NAME = "simple_ru_technical_v1"
TOKEN_PATTERN = re.compile(r"[0-9a-zа-я]+(?:-[0-9a-zа-я]+)*")


def normalize_text(text: str) -> str:
    """Привести регистр и унифицировать русские буквы ё/е."""

    return text.lower().replace("ё", "е")


def tokenize(text: str) -> list[str]:
    """Выделить слова, числа и дефисные технические идентификаторы."""

    return TOKEN_PATTERN.findall(normalize_text(text))
