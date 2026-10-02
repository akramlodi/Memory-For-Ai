import pytest

from mini_supermemory.config import ConfigError, Settings, load_settings
from mini_supermemory.llm import create_llm


def test_defaults_per_provider():
    assert Settings(llm_provider="anthropic").model == "claude-opus-5-5"
    assert Settings(llm_provider="ollama").model == "qwen2.5:7b"
    assert Settings(llm_provider="openai", llm_model="gpt-x").model == "gpt-x"


def test_missing_key_is_a_friendly_error():
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        create_llm(Settings(llm_provider="anthropic", anthropic_api_key=""))
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        create_llm(Settings(llm_provider="openai"))


def test_unknown_provider():
    with pytest.raises(ConfigError, match="one of"):
        Settings(llm_provider="nope").validate_llm()


def test_ollama_needs_no_key():
    llm = create_llm(Settings(llm_provider="ollama"))
    assert llm.name == "ollama"


def test_relative_db_path_resolves_to_project_root(monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", "data/x.db")
    s = load_settings()
    assert s.database_path.is_absolute() and s.database_path.name == "x.db"


def test_health_endpoint():
    from fastapi.testclient import TestClient

    from mini_supermemory.api import app

    r = TestClient(app).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
