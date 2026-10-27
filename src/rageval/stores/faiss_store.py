from __future__ import annotations

import numpy as np

from .base import SearchHit, VectorStore


class FaissStore(VectorStore):
    """Exact inner-product index. Vectors are L2-normalized upstream, so IP == cosine."""

    name = "faiss"

    def __init__(self, dim: int):
        super().__init__(dim)
        try:
            import faiss
        except ImportError as e:
            raise ImportError("install faiss: pip install 'rag-eval-harness[faiss]'") from e
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        self._index.add_with_ids(
            np.ascontiguousarray(vectors, dtype=np.float32), np.asarray(ids, dtype=np.int64)
        )

    def search(self, vector: np.ndarray, k: int) -> list[SearchHit]:
        query = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        scores, ids = self._index.search(query, k)
        return [SearchHit(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
