import os

# Tests never touch a real provider or download models.
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("LLM_PROVIDER", "ollama")

import pytest

from elephantus.config import Settings
from elephantus.embeddings import HashEmbedder
from elephantus.engine import MemoryEngine

from .fakes import FakeLLM


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def engine(tmp_path, fake_llm):
    settings = Settings(embedding_backend="hash", database_path=tmp_path / "test.db")
    eng = MemoryEngine(settings, embedder=HashEmbedder(), llm=fake_llm)
    yield eng
    eng.close()
