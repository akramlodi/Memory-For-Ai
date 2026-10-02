"""Text embeddings.

* ``FastEmbedder`` — a real local model (default ``BAAI/bge-small-en-v1.5``) run with
  ONNX Runtime via `fastembed`. The model (~70 MB) downloads automatically on first use.
* ``HashEmbedder`` — a dependency-free "bag of words + character n-grams" hashing
  vectoriser. Much weaker semantically, but deterministic and offline. Used in tests
  and as an escape hatch when the model cannot be downloaded.

All embedders return L2-normalised float32 vectors, so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol

import numpy as np

from .config import PROJECT_ROOT, ConfigError, Settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class FastEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self.dim = 0

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                log.info("Loading embedding model %s (first run downloads it)...", self.model_name)
                self._model = TextEmbedding(
                    model_name=self.model_name, cache_dir=str(PROJECT_ROOT / "data" / "models")
                )
            except Exception as exc:  # network errors, unknown model names, ...
                raise ConfigError(
                    f"Could not load embedding model '{self.model_name}': {exc}. "
                    "Check your internet connection for the one-time download, or set "
                    "EMBEDDING_BACKEND=hash in .env to use the offline fallback."
                ) from exc
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        vecs = np.array(list(self._load().embed(texts)), dtype=np.float32)
        self.dim = vecs.shape[1]
        return _normalise(vecs)


_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the and or but of to in on at for with is are was were be been am i im i'm my me "
    "you your it its this that these those as by from user users s".split()
)


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str):
        words = [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]
        for w in words:
            yield "w:" + w, 1.0
            padded = f"#{w}#"
            for i in range(len(padded) - 2):
                yield "c:" + padded[i : i + 3], 0.3

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for feat, weight in self._features(text):
            h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
            v[h % self.dim] += weight if (h >> 63) & 1 else -weight
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._vector(t) for t in texts]))


def create_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    return FastEmbedder(settings.embedding_model)
