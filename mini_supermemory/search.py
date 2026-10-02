"""Retrieval helpers: brute-force cosine search, FTS5 query building, Reciprocal Rank Fusion."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np

_WORD = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset(
    """a about above after again all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has
    have having he her here hers him his how i if in into is it its itself just me more most my
    no nor not now of off on once only or other our out over own same she should so some such
    than that the their them then there these they this those through to too under until up very
    was we were what when where which while who whom why will with would you your user users
    tell know think want like get got im i'm""".split()
)


def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return [(row_index, similarity)] of the k most similar rows (vectors are pre-normalised)."""
    if matrix.size == 0 or k <= 0:
        return []
    sims = matrix @ query
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def fts_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 OR-query of its content words (None if nothing usable)."""
    words = [w.lower() for w in _WORD.findall(text)]
    terms = sorted({w for w in words if w not in _STOPWORDS and len(w) > 1})
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


def reciprocal_rank_fusion(rankings: Iterable[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse several ranked id lists: score(id) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
