from __future__ import annotations

import hashlib
import re

import numpy as np

from .config import EmbedderConfig

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set(
    """a an the and or of to in on for is are was were be by with as at it that this
    from what how which who does do can if its their they we our""".split()
)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class HashingEmbedder:
    """Bag-of-words vectors via the hashing trick.

    No model download and fully deterministic, which keeps CI fast and makes
    the harness usable offline. Use sentence-transformers for real benchmarks.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, token: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "little")
        sign = 1.0 if (h >> 63) & 1 else -1.0
        return h % self.dim, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]
            # unigrams plus bigrams so word order counts for a little
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feat in features:
                idx, sign = self._bucket(feat)
                out[row, idx] += sign
        # sublinear tf
        out = np.sign(out) * np.log1p(np.abs(out))
        return _normalize(out)


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "install sentence-transformers: pip install 'rag-eval-harness[embeddings]'"
            ) from e
        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.name = model

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=32, show_progress_bar=False)
        return _normalize(np.asarray(vecs, dtype=np.float32))


def build_embedder(cfg: EmbedderConfig):
    if cfg.provider == "hashing":
        return HashingEmbedder(cfg.dim)
    if cfg.provider == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg.model)
    raise ValueError(f"unknown embedder: {cfg.provider}")
