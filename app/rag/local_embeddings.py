"""Локальные multilingual embeddings через Sentence Transformers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .models import Chunk


class EmbeddingError(ValueError):
    """Ошибка подготовки локальных векторов."""


def chunk_embedding_text(chunk: Chunk) -> str:
    """Добавить к тексту смысловые metadata, уже принадлежащие чанку."""

    return (
        f"Документ: {chunk.document_title}\n"
        f"Раздел: {chunk.section}\n\n"
        f"{chunk.text}"
    )


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Вернуть L2-нормализованную contiguous-матрицу float32."""

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise EmbeddingError(
            f"Ожидалась двумерная матрица embeddings, получена форма {matrix.shape}"
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise EmbeddingError("Матрица embeddings не должна быть пустой")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(norms.ravel() == 0)
    if zero_rows.size:
        raise EmbeddingError(
            "Невозможно нормализовать нулевые embeddings, строки: "
            + ", ".join(map(str, zero_rows.tolist()))
        )
    normalized = matrix / norms
    return np.ascontiguousarray(normalized, dtype=np.float32)


class LocalEmbeddingModel:
    """Ленивая обёртка: одна модель для документов и запросов."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Название embedding-модели не должно быть пустым")
        if batch_size <= 0:
            raise ValueError("batch_size должен быть положительным")
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: Any | None = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int | None:
        """Фактическая размерность после первого вычисления embeddings."""

        return self._dimension

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
        """Пакетно вычислить и нормализовать embeddings текстов."""

        materialized = list(texts)
        if not materialized:
            raise EmbeddingError("Список текстов для embeddings не должен быть пустым")
        empty_positions = [index for index, text in enumerate(materialized) if not text.strip()]
        if empty_positions:
            raise EmbeddingError(
                "Пустые тексты для embeddings, позиции: "
                + ", ".join(map(str, empty_positions))
            )

        raw_vectors = self._get_model().encode(
            materialized,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            precision="float32",
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        vectors = l2_normalize(np.asarray(raw_vectors, dtype=np.float32))
        dimension = int(vectors.shape[1])
        if self._dimension is None:
            self._dimension = dimension
        elif self._dimension != dimension:
            raise EmbeddingError(
                f"Размерность модели изменилась: {self._dimension} -> {dimension}"
            )
        return vectors

    def encode_query(self, query: str) -> np.ndarray:
        """Вычислить embedding запроса той же моделью и тем же способом."""

        return self.encode([query], show_progress=False)[0]

