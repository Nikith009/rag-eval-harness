from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ChunkerConfig(BaseModel):
    name: str
    strategy: Literal["fixed", "recursive", "sentence"]
    # fixed / recursive sizes are measured in words
    size: int = 300
    overlap: int = 50
    # sentence strategy
    sentences: int = 5
    sentence_overlap: int = 1

    @field_validator("overlap")
    @classmethod
    def overlap_smaller_than_size(cls, v: int, info) -> int:
        size = info.data.get("size", 300)
        if v >= size:
            raise ValueError(f"overlap ({v}) must be smaller than size ({size})")
        return v


class EmbedderConfig(BaseModel):
    provider: Literal["hashing", "sentence-transformers"] = "hashing"
    model: str = "all-MiniLM-L6-v2"
    dim: int = 512


class GeneratorConfig(BaseModel):
    provider: Literal["claude", "extractive"] = "claude"
    model: str = "claude-haiku-5-5"
    max_tokens: int = 400
    temperature: float = 0.0


class EvaluatorConfig(BaseModel):
    backend: Literal["judge", "ragas", "none"] = "judge"
    model: str = "claude-sonnet-5-5"


class PineconeConfig(BaseModel):
    index_name: str = "rag-eval"
    cloud: str = "aws"
    region: str = "us-east-1"


class RunConfig(BaseModel):
    name: str = "benchmark"
    corpus: Path
    testset: Path
    output_dir: Path = Path("results")
    cache_dir: Path = Path(".cache")
    top_k: int = 4
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    chunkers: list[ChunkerConfig]
    stores: list[Literal["memory", "faiss", "milvus", "pinecone"]] = ["memory"]
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    pinecone: PineconeConfig = Field(default_factory=PineconeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        # relative paths in the config are resolved from the working directory,
        # so run the CLI from the repo root
        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)
