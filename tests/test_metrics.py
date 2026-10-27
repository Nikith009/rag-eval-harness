import json

import pytest

from rageval.llm import parse_json
from rageval.metrics.judge import FAITHFULNESS_SYSTEM, JudgeEvaluator, average_precision
from rageval.metrics.retrieval import retrieval_metrics


def test_retrieval_metrics_basic():
    m = retrieval_metrics(["x", "a", "a", "y"], ["a"])
    assert m["hit_rate"] == 1.0
    assert m["mrr"] == 0.5
    assert m["precision_at_k"] == 0.5
    assert m["recall_at_k"] == 1.0


def test_retrieval_metrics_partial_recall_and_miss():
    m = retrieval_metrics(["a", "x"], ["a", "b"])
    assert m["recall_at_k"] == 0.5

    miss = retrieval_metrics(["x", "y"], ["a"])
    assert miss == {"hit_rate": 0.0, "mrr": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0}


def test_retrieval_metrics_without_labels():
    assert retrieval_metrics(["a"], []) == {}


def test_average_precision():
    assert average_precision([True, True]) == 1.0
    assert average_precision([False, True]) == 0.5
    assert average_precision([True, False, True]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([False, False]) == 0.0


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"score": 4}\n```') == {"score": 4}
    with pytest.raises(ValueError):
        parse_json("no json here")


class ScriptedLLM:
    """Returns a canned reply per metric, picked by the system prompt's first words."""

    model = "fake"

    def __init__(self, replies: dict[str, dict]):
        self.replies = replies
        self.calls = 0

    def complete(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        self.calls += 1
        for prefix, reply in self.replies.items():
            if system.startswith(prefix):
                return json.dumps(reply)
        raise AssertionError(f"unexpected system prompt: {system[:40]}")


def test_judge_evaluator_scores():
    llm = ScriptedLLM(
        {
            "You check whether": {
                "claims": [
                    {"claim": "a", "supported": True},
                    {"claim": "b", "supported": True},
                    {"claim": "c", "supported": False},
                ]
            },
            "You rate": {"score": 4, "reason": "mostly answers"},
            "You judge": {"verdicts": [False, True, True]},
            "You check how much": {
                "statements": [
                    {"statement": "s1", "attributed": True},
                    {"statement": "s2", "attributed": False},
                ]
            },
        }
    )
    scores = JudgeEvaluator(llm).score("q", "ref", "answer", ["c1", "c2", "c3"])

    assert scores["faithfulness"] == pytest.approx(2 / 3)
    assert scores["answer_relevancy"] == 0.75
    assert scores["context_precision"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert scores["context_recall"] == 0.5
    assert llm.calls == 4


def test_faithfulness_of_refusal_is_one():
    llm = ScriptedLLM({FAITHFULNESS_SYSTEM[:15]: {"claims": []}})
    assert JudgeEvaluator(llm).faithfulness("I don't know.", ["ctx"]) == 1.0
