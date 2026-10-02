from fastapi.testclient import TestClient

from tests.test_linking import ADIDAS, PUMA, sneaker_llm  # noqa: F401
from tests.test_retrieval import SNEAKERS


def test_chat_side_by_side(engine, sneaker_llm):  # noqa: F811
    for msg in SNEAKERS:
        engine.add(msg, "khan")
    # Echo the context back as the "answer" so we can inspect exactly what each mode saw.
    sneaker_llm.answer = lambda system, prompt: system
    out = engine.chat("What sneakers should I buy?", "khan", limit=1)
    assert out["rag"]["context"][0]["text"] == "I love Adidas sneakers"  # similarity-only baseline
    assert "Adidas sneakers" in out["rag"]["answer"]
    mem_answer = out["memory"]["answer"]
    assert PUMA in mem_answer and ADIDAS not in mem_answer
    assert "ingested" not in out


def test_chat_remember_ingests_after_answering(engine, fake_llm):
    out = engine.chat("I just adopted a cat", "khan", remember=True)
    assert out["rag"]["context"] == []  # the message did not see itself
    assert len(engine.list_documents("khan")) == 1


def test_chat_api(engine, monkeypatch):
    from elephantus import api

    monkeypatch.setattr(api, "get_engine", lambda: engine)
    r = TestClient(api.app).post("/v1/chat", json={"question": "hi?", "container_tag": "khan"})
    assert r.status_code == 200 and set(r.json()) >= {"rag", "memory"}
