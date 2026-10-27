# rag-eval-harness

[![ci](https://github.com/Nikith009/rag-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Nikith009/rag-eval-harness/actions/workflows/ci.yml) [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Nikith009/rag-eval-harness)

A small harness for answering a practical question: **which chunking strategy and vector database should this RAG system use?**

It runs every combination of chunker × vector store in a config against a labelled question set, generates answers with Claude, and scores each configuration on retrieval quality and answer quality. Results land in a CSV, a JSON summary and a Markdown report, so configurations can be compared side by side instead of picked by gut feel.

## Try it

- **Run the tests:** [CI workflow](https://github.com/Nikith009/rag-eval-harness/actions/workflows/ci.yml) → *Run workflow*. The `integration` job runs a real benchmark with sentence-transformers on memory, FAISS and Milvus and prints the report on the run page.
- **Run a benchmark with Claude:** add an `ANTHROPIC_API_KEY` repository secret, then [demo workflow](https://github.com/Nikith009/rag-eval-harness/actions/workflows/demo.yml) → *Run workflow*. Pick how many questions to use and whether to judge answers; the comparison table appears on the run page. Without the secret it runs retrieval-only.
- **In the browser:** [open a Codespace](https://codespaces.new/Nikith009/rag-eval-harness), then `rageval run -c configs/ci.yaml --offline`.

## What it measures

**Retrieval** (no LLM needed, computed from the `sources` labels in the test set)

| Metric | Meaning |
|---|---|
| `hit_rate` | at least one retrieved chunk comes from a correct source document |
| `mrr` | 1 / rank of the first relevant chunk |
| `precision_at_k` | share of the top-k chunks that come from a source document |
| `recall_at_k` | share of the source documents that appear in the top-k |

**Answer quality** (LLM-as-judge, same definitions as [RAGAS](https://arxiv.org/abs/2309.15217))

| Metric | Meaning |
|---|---|
| `faithfulness` | share of claims in the answer that the retrieved context supports |
| `answer_relevancy` | does the answer address the question |
| `context_precision` | are the relevant chunks ranked at the top |
| `context_recall` | can the context back up every statement in the reference answer |

Two judge backends are available: `judge` (built in, one Claude call per metric with JSON output) and `ragas` (the official library, using Claude through LangChain).

## How it works

```
corpus ──► chunker ──► embedder ──► vector store ──► top-k chunks ──► Claude ──► answer
              │                          │                 │                       │
        fixed / recursive /      memory / FAISS /     retrieval metrics      judge metrics
            sentence           Milvus / Pinecone
```

- Chunks are embedded once per chunker and reused across stores, so store comparisons use identical vectors.
- Every LLM response is cached on disk (`.cache/llm_responses.jsonl`), keyed by model + prompt. Re-running after changing one chunker only pays for prompts that changed, and identical retrievals from different stores cost nothing extra.
- The system prompt is marked for Anthropic prompt caching, since it is the same on every call of a metric.
- Answer generation and judging run in a thread pool (`--workers`).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
cp .env.example .env   # add ANTHROPIC_API_KEY
```

Offline smoke run (no API key, no model downloads):

```bash
rageval run -c configs/ci.yaml --offline
```

Full benchmark with Claude answers and judged metrics:

```bash
rageval run -c configs/sample.yaml
rageval run -c configs/sample.yaml --limit 5 --no-judge   # cheaper first pass
```

Each run writes to `results/<name>-<timestamp>/`:

```
report.md        ranked comparison tables
summary.csv      one row per configuration
summary.json
samples.jsonl    every question, retrieved chunks, answer and scores
usage.json       token counts per model
config.yaml      the exact config that produced the run
```

## Configuration

```yaml
name: sample-benchmark
corpus: data/sample/docs          # .md, .txt, .pdf
testset: data/sample/testset.jsonl
top_k: 4

embedder:
  provider: sentence-transformers  # or hashing
  model: all-MiniLM-L6-v2

chunkers:
  - { name: fixed-300, strategy: fixed, size: 300, overlap: 50 }       # sizes in words
  - { name: recursive-150, strategy: recursive, size: 150, overlap: 30 }
  - { name: sentence-4, strategy: sentence, sentences: 4, sentence_overlap: 1 }

stores: [faiss, milvus, pinecone]

generator: { provider: claude, model: claude-haiku-5-5 }
evaluator: { backend: judge, model: claude-sonnet-5-5 }   # judge | ragas | none
```

Test set format (one JSON object per line):

```json
{"id": "expense-02", "question": "What is the hotel limit per night in London?",
 "reference": "London is a high-cost city, so the hotel limit is 350 USD per night.",
 "sources": ["expense-policy"]}
```

`sources` are document ids: the file path relative to the corpus folder, without the extension.

### Chunking strategies

- **fixed**: sliding window of `size` words with `overlap` words shared between neighbours.
- **recursive**: splits on paragraphs, then lines, then sentences, then words, only going finer when a piece is still too long, and merges small pieces back up to `size`. Keeps paragraphs and sentences intact where possible.
- **sentence**: windows of `sentences` sentences with `sentence_overlap` shared.

### Vector stores

| Store | Notes |
|---|---|
| `memory` | numpy brute force; exact baseline |
| `faiss` | `IndexFlatIP` on normalized vectors (exact cosine) |
| `milvus` | Milvus Lite by default (local file, no server); pass a `uri` for a real cluster |
| `pinecone` | serverless index per embedding size, one namespace per run, deleted afterwards |

All backends are optional extras and imported only when used.

## Sample data

`data/sample/` holds six short policy documents for a fictional company and 18 questions, including three that need information from two documents. It is small on purpose, so the whole benchmark runs in a couple of minutes. Point `corpus` and `testset` at your own data for real comparisons.

## Development

```bash
pytest -q
ruff check . && ruff format --check .
```

CI runs lint, the test suite and the offline benchmark on Python 3.10 and 3.12, and uploads the smoke report as a build artifact.

## Layout

```
src/rageval/
  config.py          pydantic models for the YAML config
  documents.py       corpus and test set loading
  chunking.py        fixed, recursive and sentence chunkers
  embeddings.py      hashing and sentence-transformers embedders
  stores/            memory, FAISS, Milvus, Pinecone
  llm.py             Claude client, response cache, token usage
  generation.py      answer prompt, offline extractive generator
  metrics/           retrieval metrics, LLM judge, RAGAS backend
  runner.py          the benchmark loop
  report.py          CSV and Markdown output
  cli.py             `rageval` command
```
