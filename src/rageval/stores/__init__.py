from .base import SearchHit, VectorStore


def build_store(kind: str, dim: int, run_tag: str, **options) -> VectorStore:
    """Create an empty store. Backends are imported lazily so unused ones need not be installed."""
    if kind == "memory":
        from .memory import MemoryStore

        return MemoryStore(dim)
    if kind == "faiss":
        from .faiss_store import FaissStore

        return FaissStore(dim)
    if kind == "milvus":
        from .milvus_store import MilvusStore

        return MilvusStore(dim, collection=run_tag, **options)
    if kind == "pinecone":
        from .pinecone_store import PineconeStore

        return PineconeStore(dim, namespace=run_tag, **options)
    raise ValueError(f"unknown vector store: {kind}")


__all__ = ["SearchHit", "VectorStore", "build_store"]
