"""A tiny LLM provider abstraction: Anthropic, OpenAI, or Ollama.

Every provider exposes one method::

    complete(system, messages, max_tokens=..., json_mode=False, effort="low") -> str

``messages`` is a list of ``{"role": "user"|"assistant", "content": str}``.
The engine only ever needs plain text back (JSON is parsed by the caller), which
keeps the abstraction small.
"""

from __future__ import annotations

import logging
from typing import Protocol

from .config import ConfigError, Settings

log = logging.getLogger(__name__)

Message = dict[str, str]


class LLMError(Exception):
    """A user-facing error talking to the LLM provider."""


class LLM(Protocol):
    name: str
    model: str

    def complete(
        self,
        system: str,
        messages: list[Message],
        max_tokens: int = 2000,
        json_mode: bool = False,
        effort: str = "low",
    ) -> str: ...


# Models that accept `output_config.effort` / the server-side refusal fallback.
_EFFORT_PREFIXES = (
    "claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5",
    "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-6",
)
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")
_FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=3)
        self.model = model

    def complete(self, system, messages, max_tokens=2000, json_mode=False, effort="low"):
        a = self._anthropic
        kwargs: dict = {}
        if self.model.startswith(_EFFORT_PREFIXES):
            # Extraction and relation judging are simple tasks: low effort keeps them fast/cheap.
            kwargs["output_config"] = {"effort": effort}
        if self.model in _FALLBACK_MODELS:
            # Server-side refusal fallback: if a safety classifier declines, the API
            # re-runs the request on a fallback model instead of returning nothing.
            kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            kwargs["extra_body"] = {"fallbacks": "default"}
        if json_mode:
            system = system + "\n\nRespond with a single JSON object and nothing else."
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                **kwargs,
            )
        except a.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env.") from exc
        except a.NotFoundError as exc:
            raise LLMError(f"Anthropic model '{self.model}' not found. Check LLM_MODEL in .env.") from exc
        except a.RateLimitError as exc:
            raise LLMError("Anthropic rate limit reached; try again shortly.") from exc
        except a.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except a.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API (network problem?).") from exc
        if resp.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request.")
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class OpenAICompatibleLLM:
    """OpenAI, Ollama and Azure AI Foundry — all speak the OpenAI API (/v1)."""

    def __init__(self, name: str, model: str, api_key, base_url: str | None = None):
        import openai

        self._openai = openai
        self.name = name
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=2, timeout=180)

    def complete(self, system, messages, max_tokens=2000, json_mode=False, effort="low"):
        o = self._openai
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            system = system + "\n\nRespond with a single JSON object and nothing else."
        if self.name == "ollama":
            kwargs.update(max_tokens=max_tokens, temperature=0)
        else:
            # OpenAI / Azure: max_completion_tokens works for every model; reasoning models
            # (o-series, gpt-5) reject max_tokens and any non-default temperature.
            kwargs["max_completion_tokens"] = max_tokens
            if not self.model.lower().startswith(_REASONING_PREFIXES):
                kwargs["temperature"] = 0
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                **kwargs,
            )
        except o.AuthenticationError as exc:
            if self.name == "azure":
                raise LLMError("Azure rejected the credentials. Check AZURE_API_KEY (or your `az login` "
                               "identity when AZURE_USE_ENTRA_ID=true) and AZURE_ENDPOINT.") from exc
            raise LLMError("OpenAI rejected the API key. Check OPENAI_API_KEY in .env.") from exc
        except o.PermissionDeniedError as exc:
            raise LLMError(f"{self.name} denied access: {exc.message}. With Entra ID, your identity needs "
                           "a role such as 'Azure AI User' on the resource.") from exc
        except o.NotFoundError as exc:
            hint = {"ollama": f" Run `ollama pull {self.model}`.",
                    "azure": " For Azure, LLM_MODEL must be the *deployment name* shown in the Foundry portal."
                    }.get(self.name, "")
            raise LLMError(f"Model '{self.model}' not found.{hint}") from exc
        except o.APIConnectionError as exc:
            if self.name == "ollama":
                raise LLMError(
                    "Could not reach Ollama. Is it running (`ollama serve`) and is OLLAMA_BASE_URL correct?"
                ) from exc
            if self.name == "azure":
                raise LLMError("Could not reach Azure AI Foundry. Check AZURE_ENDPOINT.") from exc
            raise LLMError("Could not reach the OpenAI API (network problem?).") from exc
        except o.APIStatusError as exc:
            raise LLMError(f"{self.name} API error {exc.status_code}: {exc.message}") from exc
        return (resp.choices[0].message.content or "").strip()


def create_llm(settings: Settings) -> LLM:
    """Build the configured provider, failing with a friendly ConfigError if misconfigured."""
    settings.validate_llm()
    p = settings.llm_provider
    if p == "anthropic":
        return AnthropicLLM(settings.anthropic_api_key, settings.model)
    if p == "openai":
        return OpenAICompatibleLLM("openai", settings.model, settings.openai_api_key)
    if p == "ollama":
        return OpenAICompatibleLLM("ollama", settings.model, "ollama", settings.ollama_base_url + "/v1")
    if p == "azure":
        return OpenAICompatibleLLM("azure", settings.model, _azure_credential(settings), settings.azure_base_url)
    raise ConfigError(f"Unknown provider {p}")  # unreachable: validate_llm checks this


def _azure_credential(settings: Settings):
    """An API key, or (AZURE_USE_ENTRA_ID=true) a callable that returns fresh Entra ID tokens."""
    if not settings.azure_use_entra_id:
        return settings.azure_api_key
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except ImportError as exc:
        raise ConfigError(
            "AZURE_USE_ENTRA_ID=true needs the azure-identity package: pip install -e \".[azure]\""
        ) from exc
    return get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")


def check_llm(llm: LLM) -> str:
    """Round-trip a trivial prompt; returns the model's reply or raises LLMError."""
    return llm.complete(
        "You are a connectivity check.",
        [{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=1000,
    )
