import pytest
from fastapi.testclient import TestClient

from tests.test_linking import ADIDAS, BROKE, PUMA, sneaker_llm  # noqa: F401  (fixture reuse)

SNEAKERS = ["I love Adidas sneakers", "My Adidas broke after a month", "I'm switching to Puma"]


@pytest.fixture
def sneakers(engine, sneaker_llm):  # noqa: F811
    for msg in SNEAKERS:
        engine.add(msg, "khan")
    return engine


def test_memory_search_excludes_outdated(sneakers):
    texts = [r["text"] for r in sneakers.search("Adidas sneakers", "khan", "memories", limit=10)]
    assert ADIDAS not in texts
    assert PUMA in texts and BROKE in texts


def test_documents_mode_is_naive_baseline(sneakers):
    top = sneakers.search("Adidas sneakers", "khan", "documents", limit=1)[0]
    assert top["type"] == "chunk" and top["text"] == "I love Adidas sneakers"


def test_hybrid_mode_mixes_memories_and_chunks(sneakers):
    results = sneakers.search("Puma sneakers", "khan", "hybrid", limit=10)
    assert {r["type"] for r in results} == {"memory", "chunk"}
    assert ADIDAS not in [r["text"] for r in results if r["type"] == "memory"]


def test_keyword_search_finds_exact_term(engine, fake_llm):
    fake_llm.facts["my dog"] = [{"text": "User has a dog named Zorblax"}]
    fake_llm.facts["work"] = [{"text": "User works as a nurse"}]
    engine.add("my dog", "khan")
    engine.add("work", "khan")
    assert engine.search("Zorblax", "khan", "memories", 1)[0]["text"] == "User has a dog named Zorblax"


def test_expiry_and_simulated_time(engine, fake_llm):
    fake_llm.facts["I have an exam tomorrow"] = [
        {"text": "User has an exam tomorrow", "kind": "dynamic", "expires_in_hours": 48}]
    fake_llm.facts["I study at MIT"] = [{"text": "User studies at MIT", "kind": "static"}]
    engine.add("I have an exam tomorrow", "khan")
    engine.add("I study at MIT", "khan")
    assert "User has an exam tomorrow" in [r["text"] for r in engine.search("exam", "khan")]
    later = [r["text"] for r in engine.search("exam", "khan", time_offset_hours=72)]
    assert "User has an exam tomorrow" not in later
    assert engine.profile("khan", time_offset_hours=72)["dynamic"] == []
    statuses = {m["text"]: m["status"] for m in engine.list_memories("khan", time_offset_hours=72)}
    assert statuses["User has an exam tomorrow"] == "expired"


def test_profile_splits_static_and_dynamic(sneakers, fake_llm):
    fake_llm.facts["I study at MIT"] = [{"text": "User studies at MIT", "kind": "static"}]
    sneakers.add("I study at MIT", "khan")
    prof = sneakers.profile("khan", q="sneakers")
    assert prof["static"] == ["User studies at MIT"]
    assert prof["dynamic"] == [PUMA, BROKE]  # newest first, outdated Adidas fact excluded
    assert prof["search_results"] and all(r["type"] == "memory" for r in prof["search_results"])


def test_forget_by_content_and_id(sneakers):
    forgotten = sneakers.forget("khan", content="switching to Puma sneakers")
    assert forgotten["text"] == PUMA
    assert PUMA not in [r["text"] for r in sneakers.search("Puma", "khan", limit=10)]
    assert sneakers.forget("khan", memory_id=forgotten["id"]) is None  # already forgotten
    assert sneakers.forget("khan", content="quantum chromodynamics lecture") is None


def test_invalid_mode(engine):
    with pytest.raises(ValueError):
        engine.search("x", "khan", mode="magic")


def test_api_endpoints(sneakers, monkeypatch):
    from elephantus import api

    monkeypatch.setattr(api, "get_engine", lambda: sneakers)
    c = TestClient(api.app)
    r = c.post("/v1/search", json={"q": "sneakers", "container_tag": "khan", "mode": "memories"})
    assert r.status_code == 200 and ADIDAS not in [x["text"] for x in r.json()["results"]]
    assert c.post("/v1/profile", json={"container_tag": "khan"}).json()["dynamic"]
    assert len(c.get("/v1/containers/khan/graph").json()["edges"]) == 2
    assert len(c.get("/v1/containers/khan/log").json()) == 3
    assert c.post("/v1/search", json={"q": "x", "container_tag": "khan", "mode": "magic"}).status_code == 422
    assert c.post("/v1/forget", json={"container_tag": "khan", "content": "zzz qqq"}).status_code == 404
    paths = c.get("/openapi.json").json()["paths"]
    for p in ["/v1/add", "/v1/search", "/v1/profile", "/v1/forget", "/health"]:
        assert p in paths
