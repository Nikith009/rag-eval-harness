from __future__ import annotations

import csv
from pathlib import Path

from .metrics import QUALITY_METRICS, RETRIEVAL_METRICS


def rank_key(row: dict) -> float:
    """Sort configs by mean answer quality when judged, otherwise by retrieval quality."""
    quality = [row[m] for m in QUALITY_METRICS if row.get(m) is not None]
    if quality:
        return sum(quality) / len(quality)
    retrieval = [row[m] for m in ("recall_at_k", "mrr") if row.get(m) is not None]
    return sum(retrieval) / len(retrieval) if retrieval else 0.0


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_fmt(r.get(key)) for key, _ in columns) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def render_markdown(rows: list[dict], title: str, top_k: int) -> str:
    rows = sorted(rows, key=rank_key, reverse=True)
    judged = any(r.get(m) is not None for r in rows for m in QUALITY_METRICS)

    lines = [f"# {title}", ""]
    if rows:
        best = rows[0]
        basis = "mean answer quality" if judged else "retrieval recall and MRR"
        lines += [f"Best configuration by {basis}: **{best['config']}**.", ""]

    lines += [f"## Retrieval (top_k = {top_k})", ""]
    lines.append(
        _table(
            rows,
            [("config", "Config"), ("chunks", "Chunks"), ("avg_chunk_words", "Avg words")]
            + [(m, m) for m in RETRIEVAL_METRICS]
            + [("retrieval_ms_p50", "Search ms (p50)")],
        )
    )

    if judged:
        lines += ["", "## Answer quality", ""]
        lines.append(
            _table(
                rows,
                [("config", "Config")]
                + [(m, m) for m in QUALITY_METRICS]
                + [("generation_ms_p50", "Gen ms (p50)")],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_markdown(rows: list[dict], path: Path, title: str, top_k: int) -> None:
    path.write_text(render_markdown(rows, title, top_k), encoding="utf-8")
