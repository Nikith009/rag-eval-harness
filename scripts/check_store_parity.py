"""Fail if vector stores disagree on retrieval quality for the same chunker.

Every store does exact cosine search over identical vectors, so for each
chunker the retrieval metrics should match across memory, FAISS and Milvus.
A mismatch points at a bug in one of the store adapters.
"""

import json
import sys
from collections import defaultdict

METRICS = ["hit_rate", "mrr", "precision_at_k", "recall_at_k"]
TOLERANCE = 1e-3


def main(path: str) -> int:
    rows = json.loads(open(path).read())
    by_chunker = defaultdict(list)
    for row in rows:
        by_chunker[row["chunker"]].append(row)

    failed = False
    for chunker, group in by_chunker.items():
        for metric in METRICS:
            values = {row["store"]: row[metric] for row in group}
            spread = max(values.values()) - min(values.values())
            status = "ok" if spread <= TOLERANCE else "MISMATCH"
            failed |= status != "ok"
            print(f"{chunker:15} {metric:15} {status:9} {values}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
