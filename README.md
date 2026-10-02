# Mini-Supermemory

A small, local memory layer for AI apps (work in progress — see TASK.md).

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env    # then set ONE provider
mini-sm check           # verifies the LLM provider
mini-sm api             # REST API on http://127.0.0.1:8000 (health: /health)
```
