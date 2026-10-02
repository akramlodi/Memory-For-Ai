"""REST API: a thin FastAPI layer over the memory engine."""

from __future__ import annotations

from fastapi import FastAPI

from . import __version__
from .config import load_settings

settings = load_settings()
app = FastAPI(title="Mini-Supermemory", version=__version__)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.model,
        "embedding_backend": settings.embedding_backend,
    }
