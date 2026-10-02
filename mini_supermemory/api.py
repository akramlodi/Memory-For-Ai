"""REST API: a thin FastAPI layer over the memory engine.

Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import ConfigError, load_settings
from .engine import MemoryEngine
from .llm import LLMError

settings = load_settings()
app = FastAPI(
    title="Mini-Supermemory",
    version=__version__,
    description="A small, local memory layer for AI apps (inspired by, not affiliated with, Supermemory).",
)


@lru_cache(maxsize=1)
def get_engine() -> MemoryEngine:
    return MemoryEngine(settings)


@app.exception_handler(ValueError)
async def _value_error(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ConfigError)
async def _config_error(_: Request, exc: ConfigError):
    return JSONResponse(status_code=503, content={"detail": f"Configuration error: {exc}"})


@app.exception_handler(LLMError)
async def _llm_error(_: Request, exc: LLMError):
    return JSONResponse(status_code=502, content={"detail": f"LLM error: {exc}"})


# ----------------------------------------------------------------- schemas
class AddRequest(BaseModel):
    content: str = Field(..., examples=["I love Adidas sneakers"])
    container_tag: str = Field(..., examples=["khan"])
    metadata: dict = Field(default_factory=dict)
    time_offset_hours: float = Field(0.0, description="Pretend the content arrives this many hours from now.")


class SearchRequest(BaseModel):
    q: str = Field(..., examples=["What sneakers should I buy?"])
    container_tag: str = Field(..., examples=["khan"])
    mode: Literal["documents"] = "documents"
    limit: int = Field(5, ge=1, le=50)


# --------------------------------------------------------------- endpoints
@app.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.model,
        "embedding_backend": settings.embedding_backend,
    }


@app.post("/v1/add", tags=["memory"])
def add(req: AddRequest) -> dict:
    """Store content under a container tag (chunked + embedded for retrieval)."""
    return get_engine().add(req.content, req.container_tag, req.metadata, req.time_offset_hours)


@app.post("/v1/search", tags=["memory"])
def search(req: SearchRequest) -> dict:
    results = get_engine().search_documents(req.q, req.container_tag, req.limit)
    return {"mode": req.mode, "results": results}


@app.get("/v1/containers/{container_tag}/documents", tags=["inspect"])
def list_documents(container_tag: str) -> list[dict]:
    return get_engine().list_documents(container_tag)


@app.get("/v1/containers", tags=["inspect"])
def list_containers() -> list[str]:
    return get_engine().list_containers()


@app.delete("/v1/containers/{container_tag}", tags=["inspect"])
def reset_container(container_tag: str) -> dict:
    get_engine().reset_container(container_tag)
    return {"status": "deleted", "container_tag": container_tag}
