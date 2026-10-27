from __future__ import annotations

import os
import re
import time

import numpy as np

from .base import SearchHit, VectorStore


class PineconeStore(VectorStore):
    """Pinecone serverless. Each run writes to its own namespace and deletes it on close."""

    name = "pinecone"

    def __init__(
        self,
        dim: int,
        namespace: str,
        index_name: str = "rag-eval",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        super().__init__(dim)
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as e:
            raise ImportError("install pinecone: pip install 'rag-eval-harness[pinecone]'") from e

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")

        pc = Pinecone(api_key=api_key)
        # one index per embedding size, since an index has a fixed dimension
        name = f"{index_name}-{dim}"
        if name not in pc.list_indexes().names():
            pc.create_index(
                name=name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            while not pc.describe_index(name).status["ready"]:
                time.sleep(1)

        self._index = pc.Index(name)
        self.namespace = re.sub(r"[^0-9a-zA-Z_-]", "-", namespace)
        self._count = 0

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        records = [{"id": str(i), "values": v.tolist()} for i, v in zip(ids, vectors)]
        for start in range(0, len(records), 100):
            self._index.upsert(vectors=records[start : start + 100], namespace=self.namespace)
        self._count += len(records)
        self._wait_until_indexed()

    def _wait_until_indexed(self, timeout: float = 60.0) -> None:
        # upserts are eventually consistent; queries right after can miss vectors
        deadline = time.time() + timeout
        while time.time() < deadline:
            stats = self._index.describe_index_stats()
            ns = stats.namespaces.get(self.namespace)
            if ns and ns.vector_count >= self._count:
                return
            time.sleep(1)
        raise TimeoutError(f"pinecone namespace {self.namespace} not ready after {timeout}s")

    def search(self, vector: np.ndarray, k: int) -> list[SearchHit]:
        res = self._index.query(vector=vector.tolist(), top_k=k, namespace=self.namespace)
        return [SearchHit(int(m.id), float(m.score)) for m in res.matches]

    def close(self) -> None:
        self._index.delete(delete_all=True, namespace=self.namespace)
