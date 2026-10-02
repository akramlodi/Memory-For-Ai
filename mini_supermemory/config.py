"""Configuration loaded from environment variables / a `.env` file.

The `.env` file is searched in this order:
1. the path in ``MINI_SM_ENV_FILE`` (handy for Claude Desktop, whose working
   directory is not the project),
2. the current working directory,
3. the project root (the folder that contains the ``mini_supermemory`` package).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROVIDERS = ("anthropic", "openai", "ollama", "azure")
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "qwen2.5:7b",
    "azure": "gpt-4o",  # for Azure this is the *deployment name*
}


class ConfigError(Exception):
    """Raised when the configuration is invalid. The message is user-facing."""


def _load_env_file() -> None:
    candidates = []
    if os.getenv("MINI_SM_ENV_FILE"):
        candidates.append(Path(os.environ["MINI_SM_ENV_FILE"]))
    candidates += [Path.cwd() / ".env", PROJECT_ROOT / ".env"]
    for path in candidates:
        if path.is_file():
            # override=False: real environment variables win over the file.
            load_dotenv(path, override=False)
            return


@dataclass
class Settings:
    llm_provider: str = "anthropic"
    llm_model: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_use_entra_id: bool = False
    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    database_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "memory.db")
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    ui_port: int = 8501
    default_container_tag: str = "default"
    log_level: str = "INFO"

    @property
    def model(self) -> str:
        return self.llm_model or DEFAULT_MODELS.get(self.llm_provider, "")

    @property
    def azure_base_url(self) -> str:
        """The OpenAI-compatible v1 URL for an Azure AI Foundry resource.

        Accepts the project endpoint (https://<res>.services.ai.azure.com/api/projects/<p>),
        the resource URL, or the full .../openai/v1 URL, and normalises them all.
        """
        url = urlparse(self.azure_endpoint.strip())
        if not url.scheme or not url.netloc:
            return ""
        return f"{url.scheme}://{url.netloc}/openai/v1"

    def validate_llm(self) -> None:
        """Raise a friendly ConfigError if the LLM provider is not usable."""
        if self.llm_provider not in PROVIDERS:
            raise ConfigError(
                f"LLM_PROVIDER='{self.llm_provider}' is not supported. "
                f"Set LLM_PROVIDER to one of: {', '.join(PROVIDERS)} in your .env file."
            )
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ConfigError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty. "
                "Add your key to .env (see .env.example), or switch to LLM_PROVIDER=ollama for a keyless setup."
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ConfigError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is empty. "
                "Add your key to .env (see .env.example), or switch to LLM_PROVIDER=ollama for a keyless setup."
            )
        if self.llm_provider == "azure":
            if not self.azure_base_url:
                raise ConfigError(
                    "LLM_PROVIDER=azure but AZURE_ENDPOINT is missing or not a URL. Set it to your Azure AI "
                    "Foundry endpoint, e.g. https://<resource>.services.ai.azure.com/api/projects/<project>."
                )
            if not self.azure_api_key and not self.azure_use_entra_id:
                raise ConfigError(
                    "LLM_PROVIDER=azure needs either AZURE_API_KEY, or AZURE_USE_ENTRA_ID=true to sign in "
                    "with your Azure identity (az login) instead of a key."
                )


def load_settings() -> Settings:
    _load_env_file()
    env = os.environ.get

    db_path = Path(env("DATABASE_PATH") or "data/memory.db")
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    try:
        api_port = int(env("API_PORT") or 8000)
        ui_port = int(env("UI_PORT") or 8501)
    except ValueError as exc:
        raise ConfigError(f"API_PORT / UI_PORT must be integers ({exc}).") from exc

    backend = (env("EMBEDDING_BACKEND") or "fastembed").strip().lower()
    if backend not in ("fastembed", "hash"):
        raise ConfigError("EMBEDDING_BACKEND must be 'fastembed' or 'hash'.")

    return Settings(
        llm_provider=(env("LLM_PROVIDER") or "anthropic").strip().lower(),
        llm_model=(env("LLM_MODEL") or "").strip(),
        anthropic_api_key=(env("ANTHROPIC_API_KEY") or "").strip(),
        openai_api_key=(env("OPENAI_API_KEY") or "").strip(),
        ollama_base_url=(env("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/"),
        azure_endpoint=(env("AZURE_ENDPOINT") or "").strip(),
        azure_api_key=(env("AZURE_API_KEY") or "").strip(),
        azure_use_entra_id=(env("AZURE_USE_ENTRA_ID") or "").strip().lower() in ("1", "true", "yes"),
        embedding_backend=backend,
        embedding_model=env("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5",
        database_path=db_path,
        api_host=env("API_HOST") or "127.0.0.1",
        api_port=api_port,
        ui_port=ui_port,
        default_container_tag=env("DEFAULT_CONTAINER_TAG") or "default",
        log_level=(env("LOG_LEVEL") or "INFO").upper(),
    )


def setup_logging(level: str = "INFO") -> None:
    """Log to stderr (stdout must stay clean for the MCP stdio transport)."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
