import os

# Tests never touch a real provider or download models.
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("LLM_PROVIDER", "ollama")

import pytest

from mini_supermemory.config import Settings
from mini_supermemory.embeddings import HashEmbedder
from mini_supermemory.engine import MemoryEngine


@pytest.fixture
def engine(tmp_path):
    settings = Settings(embedding_backend="hash", database_path=tmp_path / "test.db")
    eng = MemoryEngine(settings, embedder=HashEmbedder())
    yield eng
    eng.close()
