"""Smoke-test the Streamlit UI headlessly with Streamlit's AppTest."""

from pathlib import Path

import pytest

from tests.fakes import FakeLLM
from tests.test_linking import ADIDAS, BROKE, PUMA

UI = str(Path(__file__).resolve().parent.parent / "mini_supermemory" / "ui.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from mini_supermemory import engine as engine_mod

    llm = FakeLLM(
        facts={"I love Adidas sneakers": [{"text": ADIDAS}],
               "My Adidas broke after a month": [{"text": BROKE, "kind": "dynamic"}],
               "I'm switching to Puma": [{"text": PUMA, "kind": "dynamic"}]},
        relations={BROKE: ("EXTENDS", [ADIDAS]), PUMA: ("UPDATES", [ADIDAS])},
    )
    monkeypatch.setattr(engine_mod, "create_llm", lambda settings: llm)
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "ui.db"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    import streamlit as st

    st.cache_resource.clear()
    at = AppTest.from_file(UI, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def test_ui_chat_flow(app):
    for msg in ["I love Adidas sneakers", "My Adidas broke after a month", "I'm switching to Puma"]:
        app.chat_input[0].set_value(msg).run()
        assert not app.exception
    page = " ".join(m.value for m in app.markdown)
    assert "line-through" in page and ADIDAS in page and PUMA in page
    assert any("UPDATES" in str(c.proto) for c in app.get("graphviz_chart"))
    # Reset clears the container.
    app.sidebar.button[1].click().run()
    assert not app.exception
    assert "Current memories (0)" in " ".join(m.value for m in app.markdown)
