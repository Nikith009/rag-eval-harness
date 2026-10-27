from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from .chunking import chunk_documents, word_count
from .config import RunConfig
from .documents import Chunk, TestCase, load_corpus, load_testset
from .embeddings import build_embedder
from .generation import ClaudeGenerator, ExtractiveGenerator
from .llm import ClaudeClient, ResponseCache, Usage
from .metrics import QUALITY_METRICS, RETRIEVAL_METRICS
from .metrics.judge import JudgeEvaluator
from .metrics.retrieval import retrieval_metrics
from .report import write_csv, write_markdown
from .stores import build_store


@dataclass
class SampleResult:
    config: str
    chunker: str
    store: str
    case_id: str
    question: str
    reference: str
    answer: str
    contexts: list[str]
    retrieved_docs: list[str]
    retrieval_ms: float
    generation_ms: float
    scores: dict = field(default_factory=dict)


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def summarize(samples: list[SampleResult], extra: dict[str, dict]) -> list[dict]:
    by_config: dict[str, list[SampleResult]] = {}
    for s in samples:
        by_config.setdefault(s.config, []).append(s)

    rows = []
    for name, group in by_config.items():
        row = {
            "config": name,
            "chunker": group[0].chunker,
            "store": group[0].store,
            "questions": len(group),
            **extra.get(name, {}),
            "retrieval_ms_p50": round(statistics.median(s.retrieval_ms for s in group), 2),
            "generation_ms_p50": round(statistics.median(s.generation_ms for s in group), 1),
        }
        for metric in RETRIEVAL_METRICS + QUALITY_METRICS:
            row[metric] = _mean([s.scores.get(metric) for s in group])
        rows.append(row)
    return rows


class Runner:
    def __init__(
        self,
        cfg: RunConfig,
        generator=None,
        evaluator=None,
        log: Callable[[str], None] = print,
        workers: int = 4,
    ):
        self.cfg = cfg
        self.log = log
        self.workers = workers
        self.usage = Usage()
        cache = ResponseCache(Path(cfg.cache_dir) / "llm_responses.jsonl")

        if generator is not None:
            self.generator = generator
        elif cfg.generator.provider == "extractive":
            self.generator = ExtractiveGenerator()
        else:
            llm = ClaudeClient(cfg.generator.model, cfg.generator.temperature, cache, self.usage)
            self.generator = ClaudeGenerator(llm, cfg.generator.max_tokens)

        if evaluator is not None or cfg.evaluator.backend == "none":
            self.evaluator = evaluator
        elif cfg.evaluator.backend == "judge":
            self.evaluator = JudgeEvaluator(
                ClaudeClient(cfg.evaluator.model, 0.0, cache, self.usage)
            )
        else:
            from .metrics.ragas_backend import RagasEvaluator

            self.evaluator = RagasEvaluator(cfg.evaluator.model)

    def run(self, limit: int | None = None) -> Path:
        cfg = self.cfg
        docs = load_corpus(cfg.corpus)
        cases = load_testset(cfg.testset)[:limit]
        embedder = build_embedder(cfg.embedder)
        self.log(f"{len(docs)} documents, {len(cases)} questions, embedder={embedder.name}")

        query_vecs = embedder.embed([c.question for c in cases])
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        samples: list[SampleResult] = []
        extra: dict[str, dict] = {}

        for chunker in cfg.chunkers:
            chunks = chunk_documents(docs, chunker)
            t0 = time.perf_counter()
            chunk_vecs = embedder.embed([c.text for c in chunks])
            embed_s = time.perf_counter() - t0
            avg_words = statistics.fmean(word_count(c.text) for c in chunks)

            for store_kind in cfg.stores:
                name = f"{chunker.name}/{store_kind}"
                self.log(f"[{name}] {len(chunks)} chunks (avg {avg_words:.0f} words)")

                options = cfg.pinecone.model_dump() if store_kind == "pinecone" else {}
                store = build_store(store_kind, embedder.dim, f"{stamp}-{chunker.name}", **options)
                try:
                    t0 = time.perf_counter()
                    store.add([c.chunk_id for c in chunks], chunk_vecs)
                    index_s = time.perf_counter() - t0

                    retrieved = []
                    for qv in query_vecs:
                        t0 = time.perf_counter()
                        hits = store.search(qv, cfg.top_k)
                        ms = (time.perf_counter() - t0) * 1000
                        retrieved.append(([chunks[h.id] for h in hits], ms))
                finally:
                    store.close()

                config_samples = self._answer_and_score(
                    name, chunker.name, store_kind, cases, retrieved
                )
                samples.extend(config_samples)
                extra[name] = {
                    "chunks": len(chunks),
                    "avg_chunk_words": round(avg_words),
                    "embed_s": round(embed_s, 3),
                    "index_s": round(index_s, 3),
                }

        summary = summarize(samples, extra)
        return self._write(stamp, samples, summary)

    def _answer_and_score(
        self,
        config: str,
        chunker: str,
        store: str,
        cases: list[TestCase],
        retrieved: list[tuple[list[Chunk], float]],
    ) -> list[SampleResult]:
        def work(item) -> SampleResult:
            case, (chunks, retrieval_ms) = item
            contexts = [c.text for c in chunks]
            docs = [c.doc_id for c in chunks]

            t0 = time.perf_counter()
            answer = self.generator.answer(case.question, contexts)
            gen_ms = (time.perf_counter() - t0) * 1000

            scores = retrieval_metrics(docs, case.sources)
            if hasattr(self.evaluator, "score"):
                scores.update(self.evaluator.score(case.question, case.reference, answer, contexts))

            return SampleResult(
                config=config,
                chunker=chunker,
                store=store,
                case_id=case.id,
                question=case.question,
                reference=case.reference,
                answer=answer,
                contexts=contexts,
                retrieved_docs=docs,
                retrieval_ms=retrieval_ms,
                generation_ms=gen_ms,
                scores=scores,
            )

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = list(pool.map(work, zip(cases, retrieved)))

        if hasattr(self.evaluator, "score_batch"):
            batch = self.evaluator.score_batch([asdict(r) for r in results])
            for r, s in zip(results, batch):
                r.scores.update(s)

        done = [m for m in QUALITY_METRICS if any(r.scores.get(m) is not None for r in results)]
        self.log(
            f"[{config}] scored {len(results)} answers" + (f" ({', '.join(done)})" if done else "")
        )
        return results

    def _write(self, stamp: str, samples: list[SampleResult], summary: list[dict]) -> Path:
        run_dir = Path(self.cfg.output_dir) / f"{self.cfg.name}-{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "samples.jsonl", "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(asdict(s)) + "\n")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (run_dir / "usage.json").write_text(json.dumps(asdict(self.usage), indent=2))
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(json.loads(self.cfg.model_dump_json()), sort_keys=False)
        )
        write_csv(summary, run_dir / "summary.csv")
        write_markdown(summary, run_dir / "report.md", title=self.cfg.name, top_k=self.cfg.top_k)
        self.log(f"results written to {run_dir}")
        return run_dir
