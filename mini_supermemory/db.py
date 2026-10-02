"""SQLite storage: one file, created automatically.

Tables
------
documents      raw content sent by the user
chunks         pieces of documents (+ embedding) — the RAG baseline searches these
memories       atomic facts extracted from documents (+ embedding, kind, is_latest, expiry)
memory_edges   relationships between memories (UPDATES / EXTENDS), new -> old
link_log       every linking decision (incl. NEW / DUPLICATE) for debugging and the UI
chunks_fts / memories_fts   FTS5 indexes for keyword search

Times are stored as UNIX epoch seconds (float), which makes simulated time trivial.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    container_tag TEXT NOT NULL,
    content       TEXT NOT NULL,
    metadata      TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_tag ON documents(container_tag);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    container_tag TEXT NOT NULL,
    position      INTEGER NOT NULL,
    text          TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_tag ON chunks(container_tag);

CREATE TABLE IF NOT EXISTS memories (
    id                 TEXT PRIMARY KEY,
    container_tag      TEXT NOT NULL,
    text               TEXT NOT NULL,
    kind               TEXT NOT NULL CHECK (kind IN ('static', 'dynamic')),
    is_latest          INTEGER NOT NULL DEFAULT 1,
    expires_at         REAL,
    forgotten_at       REAL,
    source_document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
    embedding          BLOB NOT NULL,
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_tag ON memories(container_tag, is_latest);

CREATE TABLE IF NOT EXISTS memory_edges (
    id            TEXT PRIMARY KEY,
    container_tag TEXT NOT NULL,
    source_id     TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id     TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation      TEXT NOT NULL CHECK (relation IN ('UPDATES', 'EXTENDS')),
    reason        TEXT,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_tag ON memory_edges(container_tag);

CREATE TABLE IF NOT EXISTS link_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    container_tag TEXT NOT NULL,
    fact          TEXT NOT NULL,
    decision      TEXT NOT NULL,
    memory_id     TEXT,
    target_ids    TEXT NOT NULL DEFAULT '[]',
    reason        TEXT,
    created_at    REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, chunk_id UNINDEXED, container_tag UNINDEXED, tokenize = 'porter unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text, memory_id UNINDEXED, container_tag UNINDEXED, tokenize = 'porter unicode61'
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        # WAL lets the API, MCP server and UI processes share the file safely.
        conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
