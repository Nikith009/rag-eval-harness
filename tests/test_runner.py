import csv
import json
from pathlib import Path

from rageval.config import RunConfig
from rageval.runner import Runner

ROOT = Path(__file__).resolve().parents[1]


class FixedJudge:
    """Stands in for the LLM judge so the end-to-end test runs offline."""

    def score(self, question, reference, answer, contexts):
        return {
            "faithfulness": 1.0,
            "answer_relevancy": 0.5,
            "context_precision": 1.0 if contexts else 0.0,
            "context_recall": 0.75,
        }


def make_config(tmp_path: Path) -> RunConfig:
    cfg = RunConfig.from_yaml(ROOT / "configs" / "ci.yaml")
    cfg.corpus = ROOT / "data" / "sample" / "docs"
    cfg.testset = ROOT / "data" / "sample" / "testset.jsonl"
    cfg.output_dir = tmp_path / "results"
    cfg.cache_dir = tmp_path / "cache"
    cfg.stores = ["memory"]
    return cfg


def test_offline_run_writes_reports(tmp_path):
    cfg = make_config(tmp_path)
    run_dir = Runner(cfg, log=lambda _: None).run()

    for name in ("samples.jsonl", "summary.csv", "summary.json", "report.md", "config.yaml"):
        assert (run_dir / name).exists(), name

    summary = json.loads((run_dir / "summary.json").read_text())
    assert {row["config"] for row in summary} == {c.name + "/memory" for c in cfg.chunkers}
    for row in summary:
        assert row["questions"] == 18
        assert 0.0 <= row["recall_at_k"] <= 1.0
        # the hashing baseline should still find the right policy most of the time
        assert row["hit_rate"] >= 0.6
        assert row["faithfulness"] is None  # no judge configured

    with open(run_dir / "summary.csv") as f:
        assert len(list(csv.DictReader(f))) == len(cfg.chunkers)

    samples = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 18 * len(cfg.chunkers)
    assert all(len(s["contexts"]) == cfg.top_k for s in samples)


def test_run_with_judge_and_limit(tmp_path):
    cfg = make_config(tmp_path)
    cfg.chunkers = cfg.chunkers[:1]
    run_dir = Runner(cfg, evaluator=FixedJudge(), log=lambda _: None).run(limit=5)

    (row,) = json.loads((run_dir / "summary.json").read_text())
    assert row["questions"] == 5
    assert row["answer_relevancy"] == 0.5
    assert row["context_recall"] == 0.75
    assert "## Answer quality" in (run_dir / "report.md").read_text()
