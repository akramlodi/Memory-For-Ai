"""The memory engine: storage, ingestion, linking and retrieval.

This module knows nothing about HTTP, MCP or the UI. Those layers construct a
``MemoryEngine`` and call its methods; every method returns plain dicts/lists
so they are trivially JSON-serialisable.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid

import numpy as np

from . import db
from .chunking import chunk_text
from .config import Settings, load_settings
from .embeddings import Embedder, create_embedder
from .search import cosine_top_k

log = logging.getLogger(__name__)

HOUR = 3600.0


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _validate_tag(container_tag: str) -> str:
    tag = (container_tag or "").strip()
    if not tag:
        raise ValueError("container_tag must be a non-empty string")
    return tag


class MemoryEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm=None,
        embedder: Embedder | None = None,
        db_path=None,
    ):
        self.settings = settings or load_settings()
        self.embedder = embedder or create_embedder(self.settings)
        self._llm = llm
        self.conn = db.connect(db_path or self.settings.database_path)
        # One lock serialises DB access: SQLite connections are not safe to share
        # between threads without it (Streamlit and FastAPI both use threads).
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def now(offset_hours: float = 0.0) -> float:
        """Current time, optionally shifted to simulate the future (for expiry demos)."""
        return time.time() + offset_hours * HOUR

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.embed(texts)

    # -------------------------------------------------------------- ingestion
    def add(self, content: str, container_tag: str, metadata: dict | None = None,
            time_offset_hours: float = 0.0) -> dict:
        """Store raw content, chunk + embed it. Returns the stored document."""
        tag = _validate_tag(container_tag)
        content = (content or "").strip()
        if not content:
            raise ValueError("content must be a non-empty string")
        created = self.now(time_offset_hours)
        doc_id = _new_id("doc")
        pieces = chunk_text(content)
        vectors = self._embed(pieces)
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO documents (id, container_tag, content, metadata, created_at) VALUES (?,?,?,?,?)",
                (doc_id, tag, content, json.dumps(metadata or {}), created),
            )
            for pos, (text, vec) in enumerate(zip(pieces, vectors)):
                chunk_id = _new_id("chk")
                self.conn.execute(
                    "INSERT INTO chunks (id, document_id, container_tag, position, text, embedding, created_at)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (chunk_id, doc_id, tag, pos, text, db.to_blob(vec), created),
                )
                self.conn.execute(
                    "INSERT INTO chunks_fts (text, chunk_id, container_tag) VALUES (?,?,?)",
                    (text, chunk_id, tag),
                )
        log.info("Added document %s to '%s' (%d chunks)", doc_id, tag, len(pieces))
        return {"id": doc_id, "container_tag": tag, "content": content, "chunks": len(pieces),
                "created_at": created}

    # -------------------------------------------------------------- retrieval
    def search_documents(self, query: str, container_tag: str, limit: int = 5) -> list[dict]:
        """RAG baseline: pure embedding similarity over document chunks."""
        tag = _validate_tag(container_tag)
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, document_id, text, embedding, created_at FROM chunks WHERE container_tag = ?",
                (tag,),
            ).fetchall()
        if not rows:
            return []
        matrix = np.stack([db.from_blob(r["embedding"]) for r in rows])
        qvec = self._embed([query])[0]
        return [
            {"type": "chunk", "id": rows[i]["id"], "document_id": rows[i]["document_id"],
             "text": rows[i]["text"], "score": round(sim, 4), "created_at": rows[i]["created_at"]}
            for i, sim in cosine_top_k(qvec, matrix, limit)
        ]

    # ------------------------------------------------------------ inspection
    def list_documents(self, container_tag: str) -> list[dict]:
        tag = _validate_tag(container_tag)
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, content, metadata, created_at FROM documents WHERE container_tag = ?"
                " ORDER BY created_at",
                (tag,),
            ).fetchall()
        return [{"id": r["id"], "content": r["content"], "metadata": json.loads(r["metadata"]),
                 "created_at": r["created_at"]} for r in rows]

    def list_containers(self) -> list[str]:
        with self._lock:
            rows = self.conn.execute("SELECT DISTINCT container_tag FROM documents ORDER BY 1").fetchall()
        return [r[0] for r in rows]

    def reset_container(self, container_tag: str) -> None:
        """Delete everything stored under a container tag."""
        tag = _validate_tag(container_tag)
        with self._lock, self.conn:
            for table in ("link_log", "memory_edges", "memories", "chunks", "documents",
                          "chunks_fts", "memories_fts"):
                self.conn.execute(f"DELETE FROM {table} WHERE container_tag = ?", (tag,))
        log.info("Reset container '%s'", tag)

    def close(self) -> None:
        self.conn.close()
