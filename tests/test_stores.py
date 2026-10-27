import numpy as np
import pytest

from rageval.embeddings import HashingEmbedder
from rageval.stores import build_store


def _corpus():
    texts = [
        "Lost laptops must be reported within one hour.",
        "Hotels cost up to 350 USD per night in London.",
        "Backups are kept for 35 days.",
        "Parental leave is 16 weeks for non-birthing parents.",
    ]
    emb = HashingEmbedder(256)
    return emb, emb.embed(texts)


def test_hashing_embedder_is_normalized_and_deterministic():
    emb = HashingEmbedder(128)
    a = emb.embed(["hello world", ""])
    b = emb.embed(["hello world", ""])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)
    assert np.linalg.norm(a[1]) == 0.0


def test_memory_store_finds_the_right_text():
    emb, vecs = _corpus()
    store = build_store("memory", emb.dim, "test")
    store.add([10, 11, 12, 13], vecs)

    hits = store.search(emb.embed(["how long are backups kept"])[0], k=2)
    assert hits[0].id == 12
    assert hits[0].score >= hits[1].score
    assert len(store.search(vecs[0], k=10)) == 4


def test_faiss_matches_memory_store():
    pytest.importorskip("faiss")
    emb, vecs = _corpus()
    mem = build_store("memory", emb.dim, "t")
    fai = build_store("faiss", emb.dim, "t")
    for s in (mem, fai):
        s.add([0, 1, 2, 3], vecs)

    query = emb.embed(["report a lost laptop"])[0]
    assert [h.id for h in mem.search(query, 3)] == [h.id for h in fai.search(query, 3)]


def test_unknown_store():
    with pytest.raises(ValueError):
        build_store("redis", 8, "t")
