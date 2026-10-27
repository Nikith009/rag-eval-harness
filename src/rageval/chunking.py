from __future__ import annotations

import re

from .config import ChunkerConfig
from .documents import Chunk, Document

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
SEPARATORS = ["\n\n", "\n", ". ", " "]


def word_count(text: str) -> int:
    return len(text.split())


def fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return out


def sentence_chunks(text: str, n: int, overlap: int) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_END.split(text) if s.strip()]
    if not sentences:
        return []
    step = max(1, n - overlap)
    out = []
    for start in range(0, len(sentences), step):
        out.append(" ".join(sentences[start : start + n]))
        if start + n >= len(sentences):
            break
    return out


def _split(text: str, size: int, separators: list[str]) -> list[str]:
    """Break text into pieces no longer than `size` words, preferring big separators."""
    if word_count(text) <= size:
        return [text.strip()] if text.strip() else []

    if not separators:
        return fixed_chunks(text, size, 0)

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split(text, size, rest)

    if sep == ". ":
        # put the period back on every sentence except the last one
        parts = [p + "." for p in parts[:-1]] + [parts[-1]]

    pieces = []
    for part in parts:
        if part.strip():
            pieces.extend(_split(part, size, rest))
    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    count = 0

    for piece in pieces:
        n = word_count(piece)
        if current and count + n > size:
            chunks.append(" ".join(current))
            # carry the last few pieces into the next chunk as overlap
            tail: list[str] = []
            tail_count = 0
            for prev in reversed(current):
                prev_n = word_count(prev)
                if tail_count + prev_n > overlap:
                    break
                tail.insert(0, prev)
                tail_count += prev_n
            current, count = tail, tail_count
        current.append(piece)
        count += n

    if current:
        chunks.append(" ".join(current))
    return chunks


def recursive_chunks(text: str, size: int, overlap: int) -> list[str]:
    return _merge(_split(text, size, SEPARATORS), size, overlap)


def chunk_documents(docs: list[Document], cfg: ChunkerConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        if cfg.strategy == "fixed":
            pieces = fixed_chunks(doc.text, cfg.size, cfg.overlap)
        elif cfg.strategy == "recursive":
            pieces = recursive_chunks(doc.text, cfg.size, cfg.overlap)
        elif cfg.strategy == "sentence":
            pieces = sentence_chunks(doc.text, cfg.sentences, cfg.sentence_overlap)
        else:
            raise ValueError(f"unknown chunking strategy: {cfg.strategy}")

        for pos, text in enumerate(pieces):
            chunks.append(Chunk(chunk_id=len(chunks), doc_id=doc.doc_id, text=text, position=pos))
    return chunks
