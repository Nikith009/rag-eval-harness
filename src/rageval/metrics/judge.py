"""LLM-as-judge versions of the four core RAGAS metrics.

The definitions follow the RAGAS paper (Es et al., 2023):

- faithfulness: share of claims in the answer that the retrieved context supports
- answer_relevancy: does the answer address the question (1-5 rating, scaled to 0-1)
- context_precision: rank-weighted precision of the retrieved chunks
- context_recall: share of reference-answer statements the context can back up

Each metric is a single Claude call that returns JSON, which keeps cost
predictable. `ragas_backend.py` runs the official library instead.
"""

from __future__ import annotations

from ..generation import format_context
from ..llm import LLM, parse_json

FAITHFULNESS_SYSTEM = """You check whether an answer is grounded in the given context.
1. Break the answer into short, standalone factual claims.
2. For each claim, decide if the context directly supports it (true) or not (false).
Citation markers like [1] are not claims. A refusal such as "I don't know" has no claims.
Return only JSON: {"claims": [{"claim": "...", "supported": true}]}"""

RELEVANCY_SYSTEM = """You rate how well an answer addresses a question, ignoring whether it is factually correct.
5 = fully and directly answers it, 3 = partially answers or adds a lot of unrelated text,
1 = does not answer it or refuses.
Return only JSON: {"score": <1-5>, "reason": "..."}"""

PRECISION_SYSTEM = """You judge retrieved passages for a question.
For each numbered passage, decide if it contains information that helps produce the reference answer.
Return only JSON with one verdict per passage, in order: {"verdicts": [true, false, ...]}"""

RECALL_SYSTEM = """You check how much of a reference answer can be found in retrieved context.
1. Break the reference answer into short, standalone statements.
2. For each statement, decide if the context contains the information (true) or not (false).
Return only JSON: {"statements": [{"statement": "...", "attributed": true}]}"""


def _ratio(items: list[dict], key: str) -> float | None:
    if not items:
        return None
    return sum(bool(item.get(key)) for item in items) / len(items)


def average_precision(verdicts: list[bool]) -> float:
    """Mean of precision@k over the ranks k where a relevant chunk appears."""
    hits = 0
    total = 0.0
    for rank, relevant in enumerate(verdicts, start=1):
        if relevant:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


class JudgeEvaluator:
    name = "judge"

    def __init__(self, llm: LLM):
        self.llm = llm

    def _ask(self, system: str, prompt: str) -> dict:
        reply = self.llm.complete(system, prompt, max_tokens=800)
        try:
            return parse_json(reply)
        except ValueError:
            # one retry with a nudge; models occasionally wrap JSON in prose
            reply = self.llm.complete(
                system, prompt + "\n\nRespond with the JSON object only.", 800
            )
            return parse_json(reply)

    def faithfulness(self, answer: str, contexts: list[str]) -> float | None:
        data = self._ask(
            FAITHFULNESS_SYSTEM, f"Context:\n{format_context(contexts)}\n\nAnswer:\n{answer}"
        )
        # no claims (e.g. "I don't know") -> nothing unfaithful was said
        return _ratio(data.get("claims", []), "supported") if data.get("claims") else 1.0

    def answer_relevancy(self, question: str, answer: str) -> float:
        data = self._ask(RELEVANCY_SYSTEM, f"Question: {question}\n\nAnswer: {answer}")
        score = min(max(float(data.get("score", 1)), 1.0), 5.0)
        return (score - 1) / 4

    def context_precision(self, question: str, reference: str, contexts: list[str]) -> float:
        data = self._ask(
            PRECISION_SYSTEM,
            f"Question: {question}\n\nReference answer: {reference}\n\n"
            f"Passages:\n{format_context(contexts)}",
        )
        verdicts = [bool(v) for v in data.get("verdicts", [])][: len(contexts)]
        return average_precision(verdicts)

    def context_recall(self, reference: str, contexts: list[str]) -> float | None:
        data = self._ask(
            RECALL_SYSTEM,
            f"Context:\n{format_context(contexts)}\n\nReference answer:\n{reference}",
        )
        return _ratio(data.get("statements", []), "attributed")

    def score(self, question: str, reference: str, answer: str, contexts: list[str]) -> dict:
        return {
            "faithfulness": self.faithfulness(answer, contexts),
            "answer_relevancy": self.answer_relevancy(question, answer),
            "context_precision": self.context_precision(question, reference, contexts),
            "context_recall": self.context_recall(reference, contexts),
        }
