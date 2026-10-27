from __future__ import annotations

import re

from .chunking import SENTENCE_END
from .embeddings import STOPWORDS, TOKEN
from .llm import LLM

ANSWER_SYSTEM = """You answer questions using only the provided context passages.
Rules:
- Use only facts stated in the context. Do not add outside knowledge.
- If the context does not contain the answer, reply exactly: I don't know based on the provided documents.
- Be concise: one to three sentences.
- Cite the passages you used with their numbers in square brackets, e.g. [1] or [2][3]."""


def format_context(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))


class ClaudeGenerator:
    def __init__(self, llm: LLM, max_tokens: int = 400):
        self.llm = llm
        self.max_tokens = max_tokens
        self.name = llm.model

    def answer(self, question: str, contexts: list[str]) -> str:
        prompt = f"Context:\n{format_context(contexts)}\n\nQuestion: {question}"
        return self.llm.complete(ANSWER_SYSTEM, prompt, self.max_tokens).strip()


class ExtractiveGenerator:
    """Offline stand-in for an LLM: returns the context sentences that best overlap the question.

    Handy for smoke tests and for comparing retrieval settings without an API key.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS}

    def answer(self, question: str, contexts: list[str]) -> str:
        q = self._terms(question)
        scored = []
        for ctx_idx, ctx in enumerate(contexts):
            for sent in SENTENCE_END.split(ctx):
                sent = re.sub(r"\s+", " ", sent).strip()
                if not sent:
                    continue
                overlap = len(q & self._terms(sent))
                if overlap:
                    # small bonus for higher-ranked passages
                    scored.append((overlap - 0.01 * ctx_idx, sent))
        if not scored:
            return "I don't know based on the provided documents."
        scored.sort(key=lambda x: -x[0])
        return " ".join(s for _, s in scored[: self.max_sentences])
