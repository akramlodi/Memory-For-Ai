import numpy as np

from mini_supermemory.chunking import chunk_text
from mini_supermemory.embeddings import HashEmbedder
from mini_supermemory.search import fts_query, reciprocal_rank_fusion


def test_chunking_short_and_long():
    assert chunk_text("hello") == ["hello"]
    long = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = chunk_text(long, max_chars=200)
    assert len(chunks) > 5 and all(len(c) <= 200 for c in chunks)
    assert chunk_text("   ") == []


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder()
    a, b = e.embed(["I love Adidas sneakers", "I love Adidas sneakers"])
    assert np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5
    sims = e.embed(["sneakers", "Adidas sneakers", "tax return deadline"]) @ e.embed(["sneakers"])[0]
    assert sims[1] > sims[2]


def test_add_and_semantic_search(engine):
    engine.add("I love Adidas sneakers", "khan")
    engine.add("My tax return is due in April", "khan")
    results = engine.search_documents("sneakers", "khan", limit=2)
    assert results[0]["text"] == "I love Adidas sneakers"
    assert results[0]["type"] == "chunk"


def test_container_tags_isolate_data(engine):
    engine.add("I love Adidas sneakers", "khan")
    engine.add("I love Nike sneakers", "work")
    texts = [r["text"] for r in engine.search_documents("sneakers", "khan", limit=10)]
    assert texts == ["I love Adidas sneakers"]
    assert engine.search_documents("sneakers", "nobody") == []
    assert set(engine.list_containers()) == {"khan", "work"}


def test_reset_container(engine):
    engine.add("I love Adidas sneakers", "khan")
    engine.add("I love Nike sneakers", "work")
    engine.reset_container("khan")
    assert engine.list_documents("khan") == []
    assert len(engine.list_documents("work")) == 1


def test_empty_input_rejected(engine):
    import pytest

    with pytest.raises(ValueError):
        engine.add("  ", "khan")
    with pytest.raises(ValueError):
        engine.add("text", "")


def test_fts_query_and_rrf():
    assert fts_query("What sneakers should I buy?") == '"buy" OR "sneakers"'
    assert fts_query("what is it?") is None
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]])
    assert [i for i, _ in fused][:2] in (["a", "b"], ["b", "a"])
    assert fused[-1][0] == "c"


def test_api_add_and_search(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mini_supermemory import api
    from mini_supermemory.config import Settings
    from mini_supermemory.engine import MemoryEngine

    eng = MemoryEngine(Settings(embedding_backend="hash"), embedder=HashEmbedder(), db_path=tmp_path / "a.db")
    monkeypatch.setattr(api, "get_engine", lambda: eng)
    client = TestClient(api.app)
    assert client.post("/v1/add", json={"content": "I love Adidas sneakers", "container_tag": "khan"}).status_code == 200
    r = client.post("/v1/search", json={"q": "sneakers", "container_tag": "khan", "mode": "documents"})
    assert r.json()["results"][0]["text"] == "I love Adidas sneakers"
    assert client.post("/v1/add", json={"content": "", "container_tag": "khan"}).status_code == 422
