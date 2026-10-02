import pytest

from elephantus.config import ConfigError, Settings, load_settings
from elephantus.llm import create_llm

PROJECT = "https://jal-resource.services.ai.azure.com/api/projects/jal"
V1 = "https://jal-resource.services.ai.azure.com/openai/v1"


@pytest.mark.parametrize("endpoint", [PROJECT, V1, V1 + "/", "https://jal-resource.services.ai.azure.com"])
def test_endpoint_normalised_to_openai_v1(endpoint):
    assert Settings(llm_provider="azure", azure_endpoint=endpoint).azure_base_url == V1


def test_azure_validation_errors():
    with pytest.raises(ConfigError, match="AZURE_ENDPOINT"):
        Settings(llm_provider="azure", azure_api_key="k").validate_llm()
    with pytest.raises(ConfigError, match="AZURE_API_KEY"):
        Settings(llm_provider="azure", azure_endpoint=PROJECT).validate_llm()


def test_azure_client_uses_v1_url_key_and_deployment(monkeypatch):
    llm = create_llm(Settings(llm_provider="azure", azure_endpoint=PROJECT, azure_api_key="secret",
                              llm_model="my-gpt4o-deployment"))
    assert llm.name == "azure" and llm.model == "my-gpt4o-deployment"
    assert str(llm.client.base_url).rstrip("/") == V1 and llm.client.api_key == "secret"

    seen = {}

    class Choice:
        message = type("M", (), {"content": " OK "})

    def fake_create(**kwargs):
        seen.update(kwargs)
        return type("R", (), {"choices": [Choice]})

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)
    assert llm.complete("sys", [{"role": "user", "content": "hi"}], max_tokens=50) == "OK"
    assert seen["model"] == "my-gpt4o-deployment"
    assert seen["max_completion_tokens"] == 50 and seen["temperature"] == 0 and "max_tokens" not in seen


def test_reasoning_deployments_skip_temperature(monkeypatch):
    llm = create_llm(Settings(llm_provider="azure", azure_endpoint=PROJECT, azure_api_key="k", llm_model="gpt-5-mini"))
    seen = {}
    monkeypatch.setattr(llm.client.chat.completions, "create",
                        lambda **kw: seen.update(kw) or type("R", (), {"choices": [type("C", (), {
                            "message": type("M", (), {"content": "x"})})]}))
    llm.complete("sys", [{"role": "user", "content": "hi"}])
    assert "temperature" not in seen


def test_entra_id_without_package_is_friendly(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_azure(name, *a, **k):
        if name.startswith("azure"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_azure)
    with pytest.raises(ConfigError, match="azure-identity"):
        create_llm(Settings(llm_provider="azure", azure_endpoint=PROJECT, azure_use_entra_id=True))


def test_azure_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_ENDPOINT", PROJECT)
    monkeypatch.setenv("AZURE_USE_ENTRA_ID", "true")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    s = load_settings()
    assert s.azure_use_entra_id and s.model == "gpt-4o" and s.azure_base_url == V1
