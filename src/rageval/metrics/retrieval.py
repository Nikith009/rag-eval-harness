"""Retrieval metrics computed against the `sources` labels in the test set.

A retrieved chunk counts as relevant when it comes from one of the documents
listed as a source for that question. These need no LLM, so they are cheap
to compute for every configuration.
"""

from __future__ import annotations


def retrieval_metrics(retrieved_docs: list[str], relevant_docs: list[str]) -> dict[str, float]:
    relevant = set(relevant_docs)
    if not relevant:
        return {}

    k = len(retrieved_docs)
    hits = [doc in relevant for doc in retrieved_docs]

    first_hit = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
    found = {doc for doc in retrieved_docs if doc in relevant}

    return {
        "hit_rate": 1.0 if first_hit else 0.0,
        "mrr": 1.0 / first_hit if first_hit else 0.0,
        "precision_at_k": sum(hits) / k if k else 0.0,
        "recall_at_k": len(found) / len(relevant),
    }
