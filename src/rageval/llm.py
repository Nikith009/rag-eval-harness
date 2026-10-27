from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class LLM(Protocol):
    model: str

    def complete(self, system: str, prompt: str, max_tokens: int = 512) -> str: ...


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, inp: int, out: int, cache_read: int) -> None:
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        self.cache_read_tokens += cache_read
        m = self.by_model.setdefault(model, {"input": 0, "output": 0})
        m["input"] += inp
        m["output"] += out


class ResponseCache:
    """Append-only JSONL cache keyed by a hash of (model, system, prompt, params).

    Re-running a benchmark after changing one setting only pays for the
    prompts that actually changed.
    """

    def __init__(self, path: Path | None):
        self.path = path
        self._data: dict[str, str] = {}
        self._lock = threading.Lock()
        if path and path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self._data[row["key"]] = row["text"]

    @staticmethod
    def key(*parts: object) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, text: str) -> None:
        with self._lock:
            self._data[key] = text
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "text": text}) + "\n")


class ClaudeClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        cache: ResponseCache | None = None,
        usage: Usage | None = None,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "install the Anthropic SDK: pip install 'rag-eval-harness[claude]'"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")

        # the SDK already retries 429s and 5xx with backoff
        self._client = anthropic.Anthropic(max_retries=5)
        self.model = model
        self.temperature = temperature
        self.cache = cache or ResponseCache(None)
        self.usage = usage or Usage()

    def complete(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        key = ResponseCache.key(self.model, system, prompt, max_tokens, self.temperature)
        cached = self.cache.get(key)
        if cached is not None:
            self.usage.cached_calls += 1
            return cached

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            # the system prompt is identical across every call of a metric, so mark it
            # for prompt caching (applies once it is past the model's minimum length)
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        u = resp.usage
        self.usage.add(
            self.model,
            u.input_tokens,
            u.output_tokens,
            getattr(u, "cache_read_input_tokens", 0) or 0,
        )
        self.cache.put(key, text)
        return text


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates ```json fences)."""
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(match.group(0))
