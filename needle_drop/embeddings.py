"""Text embeddings.

Default model: nomic-embed-text-v1.5 (Apache-2.0, 768-d, Matryoshka) run locally through
llama.cpp from a GGUF file, so no API key or GPU is needed. nomic expects task prefixes:
"search_query: " for queries and "search_document: " for catalog entries.

We truncate to 256 dims (Matryoshka) to keep the index small enough to ship to the browser.
"""
from __future__ import annotations

import hashlib
import os
import threading
from functools import lru_cache

import numpy as np

from . import config

DIM = int(os.getenv("NEEDLE_EMBED_DIM", "256"))
GGUF = os.getenv("NEEDLE_EMBED_GGUF", str(config.MODELS / "nomic-embed-v1.5-f16.gguf"))


class Embedder:
    name = "nomic-embed-text-v1.5"

    def __init__(self, path: str = GGUF, dim: int = DIM):
        from llama_cpp import Llama

        self.dim = dim
        self._lock = threading.Lock()
        self._m = Llama(path, embedding=True, verbose=False, n_ctx=2048, n_threads=os.cpu_count())
        self._cache: dict[str, np.ndarray] = {}

    def _embed_one(self, text: str) -> np.ndarray:
        key = hashlib.sha1(text.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            v = np.asarray(self._m.embed(text), dtype=np.float32)
        if v.ndim > 1:
            v = v.mean(0)
        v = v[: self.dim]
        v /= np.linalg.norm(v) + 1e-9
        if len(self._cache) < 50_000:
            self._cache[key] = v
        return v

    def query(self, text: str) -> np.ndarray:
        return self._embed_one("search_query: " + text)

    def documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed_one("search_document: " + t) for t in texts])


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
