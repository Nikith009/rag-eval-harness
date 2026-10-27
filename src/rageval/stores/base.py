from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchHit:
    id: int
    score: float


class VectorStore(ABC):
    name: str = "base"

    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    def add(self, ids: list[int], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int) -> list[SearchHit]: ...

    def close(self) -> None:
        """Release resources / delete temporary data. Optional."""
