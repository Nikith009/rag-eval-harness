from __future__ import annotations

import numpy as np

from .base import SearchHit, VectorStore


class MemoryStore(VectorStore):
    """Brute-force cosine search in numpy. Baseline and reference for the other stores."""

    name = "memory"

    def __init__(self, dim: int):
        super().__init__(dim)
        self._ids = np.empty(0, dtype=np.int64)
        self._vectors = np.empty((0, dim), dtype=np.float32)

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        self._ids = np.concatenate([self._ids, np.asarray(ids, dtype=np.int64)])
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])

    def search(self, vector: np.ndarray, k: int) -> list[SearchHit]:
        if len(self._ids) == 0:
            return []
        scores = self._vectors @ vector.astype(np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [SearchHit(int(self._ids[i]), float(scores[i])) for i in top]
