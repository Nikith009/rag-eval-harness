from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import RunConfig


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import Runner

    cfg = RunConfig.from_yaml(args.config)
    if args.stores:
        cfg.stores = args.stores.split(",")
    if args.offline:
        cfg.generator.provider = "extractive"
        cfg.evaluator.backend = "none"
    elif args.no_judge:
        cfg.evaluator.backend = "none"

    run_dir = Runner(cfg, workers=args.workers).run(limit=args.limit)
    print()
    print((run_dir / "report.md").read_text())
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import render_markdown

    run_dir = Path(args.run_dir)
    rows = json.loads((run_dir / "summary.json").read_text())
    print(render_markdown(rows, title=run_dir.name, top_k=args.top_k))
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="rageval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="benchmark every chunker x store combination in a config")
    run.add_argument("-c", "--config", required=True, help="path to a YAML config")
    run.add_argument("--stores", help="comma-separated override, e.g. memory,faiss")
    run.add_argument("--limit", type=int, help="only use the first N questions")
    run.add_argument("--workers", type=int, default=4, help="parallel LLM calls")
    run.add_argument("--no-judge", action="store_true", help="skip LLM-judged metrics")
    run.add_argument(
        "--offline",
        action="store_true",
        help="no API calls: extractive answers, retrieval metrics only",
    )
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="re-render report.md for a finished run")
    report.add_argument("run_dir")
    report.add_argument("--top-k", type=int, default=4)
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
