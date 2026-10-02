"""Optional end-to-end test against the real configured LLM provider.

Run with:  ELEPHANTUS_LIVE_TESTS=1 pytest -m live
(uses your .env provider; the hash embedder keeps it download-free)
"""

import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("ELEPHANTUS_LIVE_TESTS") != "1", reason="set ELEPHANTUS_LIVE_TESTS=1 to run"),
]


def test_live_sneaker_sequence(tmp_path):
    from elephantus.config import load_settings
    from elephantus.engine import MemoryEngine

    settings = load_settings()
    engine = MemoryEngine(settings, db_path=tmp_path / "live.db")
    for msg in ["I love Adidas sneakers", "My Adidas broke after a month", "I'm switching to Puma"]:
        engine.add(msg, "live")
    mems = engine.list_memories("live")
    current = " ".join(m["text"].lower() for m in mems if m["status"] == "current")
    outdated = [m["text"].lower() for m in mems if m["status"] == "outdated"]
    assert "puma" in current
    assert any("adidas" in t and "love" in t for t in outdated)
    assert any(e["relation"] == "UPDATES" for e in engine.graph("live")["edges"])
