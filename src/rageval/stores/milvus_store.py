from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .base import SearchHit, VectorStore


class MilvusStore(VectorStore):
    """Milvus via pymilvus.

    By default this uses Milvus Lite (a local file), so no server is needed.
    Point `uri` at a running Milvus instance (e.g. http://localhost:19530) to
    benchmark the real thing.
    """

    name = "milvus"

    def __init__(self, dim: int, collection: str, uri: str | None = None, token: str = ""):
        super().__init__(dim)
        try:
            from pymilvus import MilvusClient
        except ImportError as e:
            raise ImportError("install pymilvus: pip install 'rag-eval-harness[milvus]'") from e

        if uri is None:
            Path(".cache").mkdir(exist_ok=True)
            uri = ".cache/milvus_lite.db"

        # collection names: letters, digits and underscores only
        self.collection = "c_" + re.sub(r"[^0-9a-zA-Z_]", "_", collection)
        self._client = MilvusClient(uri=uri, token=token)
        if self._client.has_collection(self.collection):
            self._client.drop_collection(self.collection)
        self._client.create_collection(
            collection_name=self.collection,
            dimension=dim,
            metric_type="COSINE",
            auto_id=False,
        )

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        rows = [{"id": int(i), "vector": v.tolist()} for i, v in zip(ids, vectors)]
        for start in range(0, len(rows), 500):
            self._client.insert(self.collection, rows[start : start + 500])

    def search(self, vector: np.ndarray, k: int) -> list[SearchHit]:
        result = self._client.search(
            collection_name=self.collection,
            data=[vector.tolist()],
            limit=k,
        )
        return [SearchHit(int(hit["id"]), float(hit["distance"])) for hit in result[0]]

    def close(self) -> None:
        self._client.drop_collection(self.collection)
        self._client.close()
