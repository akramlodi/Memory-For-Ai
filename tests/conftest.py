import os

# Tests never touch a real provider or download models.
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("LLM_PROVIDER", "ollama")
