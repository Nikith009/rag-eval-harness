from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt"}


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    text: str
    position: int  # order of the chunk inside its document


@dataclass
class TestCase:
    question: str
    reference: str
    sources: list[str]  # doc ids that contain the answer
    id: str = ""


def load_corpus(folder: Path) -> list[Document]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"corpus folder not found: {folder}")

    docs = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue
        doc_id = path.relative_to(folder).with_suffix("").as_posix()
        docs.append(Document(doc_id=doc_id, text=text, metadata={"path": str(path)}))

    if not docs:
        raise ValueError(f"no .md, .txt or .pdf files found in {folder}")
    return docs


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("PDF support needs pypdf: pip install 'rag-eval-harness[pdf]'") from e
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_testset(path: Path) -> list[TestCase]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sources = row.get("sources", [])
            if isinstance(sources, str):
                sources = [sources]
            cases.append(
                TestCase(
                    id=row.get("id") or f"q{i}",
                    question=row["question"],
                    reference=row["reference"],
                    sources=sources,
                )
            )
    return cases
