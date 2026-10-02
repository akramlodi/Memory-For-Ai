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
from .extraction import extract_facts
from .linking import Candidate, judge_relation
from .llm import LLMError, create_llm
from .search import cosine_top_k

log = logging.getLogger(__name__)

HOUR = 3600.0
LINK_CANDIDATES = 5  # how many similar current memories the relation judge sees

# SQL condition for "current" memories: latest version, not forgotten, not expired at :now.
CURRENT = "is_latest = 1 AND forgotten_at IS NULL AND (expires_at IS NULL OR expires_at > :now)"


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

    @property
    def llm(self):
        """The LLM is created lazily, so storage/search work even before a provider is configured."""
        if self._llm is None:
            self._llm = create_llm(self.settings)
        return self._llm

    @staticmethod
    def _memory_dict(r) -> dict:
        return {
            "type": "memory", "id": r["id"], "text": r["text"], "kind": r["kind"],
            "is_latest": bool(r["is_latest"]), "expires_at": r["expires_at"],
            "forgotten_at": r["forgotten_at"], "source_document_id": r["source_document_id"],
            "created_at": r["created_at"],
        }

    # -------------------------------------------------------------- ingestion
    def add(self, content: str, container_tag: str, metadata: dict | None = None,
            time_offset_hours: float = 0.0, extract: bool = True) -> dict:
        """Store raw content (chunked + embedded) and, if `extract`, turn it into linked memories.

        Returns the stored document plus one entry per extracted fact describing the
        linking decision (NEW / UPDATES / EXTENDS / DUPLICATE).
        """
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
        result = {"id": doc_id, "container_tag": tag, "content": content, "chunks": len(pieces),
                  "created_at": created, "memories": []}
        if extract:
            try:
                result["memories"] = self._ingest_memories(content, tag, doc_id, created)
            except LLMError as exc:
                # The document is safely stored; report the problem instead of failing the call.
                log.error("Memory extraction failed for %s: %s", doc_id, exc)
                result["error"] = f"Memory extraction failed: {exc}"
        return result

    # ------------------------------------------------------- memory pipeline
    def _ingest_memories(self, content: str, tag: str, doc_id: str, now_ts: float) -> list[dict]:
        known = [m["text"] for m in self._current_memory_rows(tag, now_ts)][-15:]
        facts = extract_facts(self.llm, content, now_ts, known)
        log.info("Extracted %d fact(s) from %s", len(facts), doc_id)
        # Facts are linked one at a time so later facts can relate to earlier ones.
        return [self._link_and_store(f, tag, doc_id, now_ts) for f in facts]

    def _current_memory_rows(self, tag: str, now_ts: float) -> list:
        with self._lock:
            return self.conn.execute(
                f"SELECT * FROM memories WHERE container_tag = :tag AND {CURRENT} ORDER BY created_at",
                {"tag": tag, "now": now_ts},
            ).fetchall()

    def _link_and_store(self, fact, tag: str, doc_id: str, now_ts: float) -> dict:
        vec = self._embed([fact.text])[0]
        rows = self._current_memory_rows(tag, now_ts)
        candidates: list[Candidate] = []
        if rows:
            matrix = np.stack([db.from_blob(r["embedding"]) for r in rows])
            candidates = [Candidate(rows[i]["id"], rows[i]["text"], sim)
                          for i, sim in cosine_top_k(vec, matrix, LINK_CANDIDATES)]
        judgement = judge_relation(self.llm, fact.text, candidates)
        expires_at = now_ts + fact.expires_in_hours * HOUR if fact.expires_in_hours else None

        memory_id = None
        with self._lock, self.conn:
            if judgement.relation != "DUPLICATE":
                memory_id = _new_id("mem")
                self.conn.execute(
                    "INSERT INTO memories (id, container_tag, text, kind, is_latest, expires_at,"
                    " source_document_id, embedding, created_at, updated_at) VALUES (?,?,?,?,1,?,?,?,?,?)",
                    (memory_id, tag, fact.text, fact.kind, expires_at, doc_id, db.to_blob(vec), now_ts, now_ts),
                )
                self.conn.execute(
                    "INSERT INTO memories_fts (text, memory_id, container_tag) VALUES (?,?,?)",
                    (fact.text, memory_id, tag),
                )
                for target in judgement.target_ids:
                    if judgement.relation in ("UPDATES", "EXTENDS"):
                        self.conn.execute(
                            "INSERT INTO memory_edges (id, container_tag, source_id, target_id, relation,"
                            " reason, created_at) VALUES (?,?,?,?,?,?,?)",
                            (_new_id("edge"), tag, memory_id, target, judgement.relation, judgement.reason, now_ts),
                        )
                    if judgement.relation == "UPDATES":
                        # Old facts are kept for history but are no longer current.
                        self.conn.execute(
                            "UPDATE memories SET is_latest = 0, updated_at = ? WHERE id = ?", (now_ts, target)
                        )
            self.conn.execute(
                "INSERT INTO link_log (container_tag, fact, decision, memory_id, target_ids, reason, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (tag, fact.text, judgement.relation, memory_id, json.dumps(judgement.target_ids),
                 judgement.reason, now_ts),
            )
        log.info("Fact %r -> %s %s (%s)", fact.text, judgement.relation, judgement.target_ids, judgement.reason)
        return {"fact": fact.text, "kind": fact.kind, "expires_at": expires_at, "decision": judgement.relation,
                "memory_id": memory_id, "target_ids": judgement.target_ids, "reason": judgement.reason}

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

    def list_memories(self, container_tag: str, include_outdated: bool = True,
                      time_offset_hours: float = 0.0) -> list[dict]:
        """All memories (newest first) with a `status`: current | outdated | expired | forgotten."""
        tag = _validate_tag(container_tag)
        now_ts = self.now(time_offset_hours)
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE container_tag = ? ORDER BY created_at DESC", (tag,)
            ).fetchall()
        out = []
        for r in rows:
            m = self._memory_dict(r)
            if r["forgotten_at"] is not None:
                m["status"] = "forgotten"
            elif not r["is_latest"]:
                m["status"] = "outdated"
            elif r["expires_at"] is not None and r["expires_at"] <= now_ts:
                m["status"] = "expired"
            else:
                m["status"] = "current"
            if include_outdated or m["status"] == "current":
                out.append(m)
        return out

    def graph(self, container_tag: str, time_offset_hours: float = 0.0) -> dict:
        """Nodes (all memories with status) and edges (UPDATES / EXTENDS) for visualisation."""
        tag = _validate_tag(container_tag)
        with self._lock:
            edges = self.conn.execute(
                "SELECT source_id, target_id, relation, reason, created_at FROM memory_edges"
                " WHERE container_tag = ? ORDER BY created_at", (tag,)
            ).fetchall()
        return {"nodes": self.list_memories(tag, True, time_offset_hours), "edges": [dict(e) for e in edges]}

    def link_log(self, container_tag: str, limit: int = 100) -> list[dict]:
        tag = _validate_tag(container_tag)
        with self._lock:
            rows = self.conn.execute(
                "SELECT fact, decision, memory_id, target_ids, reason, created_at FROM link_log"
                " WHERE container_tag = ? ORDER BY id DESC LIMIT ?", (tag, limit)
            ).fetchall()
        return [{**dict(r), "target_ids": json.loads(r["target_ids"])} for r in rows]

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
