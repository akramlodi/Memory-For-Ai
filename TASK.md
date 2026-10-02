# Elephantus — Implementation Brief for Claude Code

> **How to use this file:** This is the project brief. Read it fully before writing any code.
> Implement the project **phase by phase, in order**. Finish each phase, verify its acceptance
> criteria, and commit before moving to the next. Implementation details (exact libraries,
> file layout, prompts, thresholds, function names) are **your call** — choose what is simplest
> and most reliable, and explain notable choices in the README.

---

## 1. Goal

Build **Elephantus**: a simple, working, locally-runnable memory layer for AI applications.

The project must **demonstrate** the core idea clearly:

> **RAG finds similar text. Memory tracks what is *currently true* about a user.**

It is a portfolio project, so priorities are:

1. **It works end-to-end** on a single laptop with minimal setup.
2. **It demonstrates** the key behaviours visibly (UI + Claude Desktop via MCP).
3. **It proves** them with a small, reproducible evaluation (RAG baseline vs. memory).
4. **It is clean and well documented** — easy for a reviewer to clone and run.

Simplicity beats completeness. When in doubt, choose the smaller design.

---

## 2. Reference Material (read before starting)

Useful background:

| Resource | Why it matters |
|---|---|
| LongMemEval: https://github.com/xiaowu0162/LongMemEval | Benchmark categories (knowledge updates, temporal reasoning) to borrow ideas from |
| MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk | For building the MCP server |
| MCP + Claude Desktop setup: https://modelcontextprotocol.io/quickstart/user | How users connect a local MCP server |

---

## 3. Core Concepts to Implement

### 3.1 Documents vs Memories

- **Documents** = raw content the user sends (messages, notes). Stored and chunked for RAG-style retrieval.
- **Memories** = short, atomic facts **extracted** from documents by an LLM (e.g. "User is switching to Puma"). Tied to a user/project, and they evolve over time.

### 3.2 Container tags

Every document and memory belongs to a `container_tag` (e.g. `khan`, `work`). Memories from different tags must never mix.

### 3.3 Memory relationships

When a new fact arrives, compare it to existing **current** memories in the same container and classify it as:

| Label | Meaning | Effect |
|---|---|---|
| `NEW` | Unrelated to anything stored | Insert |
| `UPDATES` | Replaces/contradicts an older fact | Insert; mark old memory `is_latest = false`; record an edge |
| `EXTENDS` | Adds detail to an older fact | Insert; both remain current; record an edge |
| `DUPLICATE` | Already known | Skip |

Suggested approach: embeddings to shortlist similar candidates cheaply → LLM to judge the relationship. Old memories are **kept for history**, but normal search returns only current ones.

### 3.4 Memory kinds

- `static` — long-term facts (where someone studies, stable preferences).
- `dynamic` — recent/ongoing context (current project, what they're doing this week).

### 3.5 Forgetting

Time-bound facts ("I have an exam tomorrow") get an expiry timestamp at extraction time and are excluded from retrieval once expired. Support a **simulated "now"** (e.g. a time-offset parameter) so expiry can be demonstrated without waiting.

### 3.6 Hybrid search

Combine semantic (embedding) search with keyword search, and merge results with a rank-based fusion method (e.g. Reciprocal Rank Fusion). Search over current, unexpired memories; optionally also over document chunks.

### 3.7 Profile

A single call returning:
- `static`: current static memories
- `dynamic`: recent current dynamic memories
- optionally, search results for a query `q`

---

## 4. Architecture: One Engine, Three Entry Points

```
                ┌──────────────────────────┐
  Demo UI  ─────┤                          │
                │      Memory Engine       │──► LLM provider (extraction + relation judge + chat)
  REST API ─────┤   (storage, ingestion,   │
                │    linking, retrieval)   │──► Local embedding model
  MCP server ───┤                          │
                └──────────────────────────┘
                         │
                     Local database (single file)
```

- The **engine** is a plain library with no knowledge of HTTP or MCP.
- The **REST API**, **MCP server**, and **UI** are thin layers over the same engine and same database.

### Suggested stack (you may substitute with justification)

- Python 3.11+
- FastAPI for the REST API
- SQLite (single file) for storage; SQLite FTS5 for keyword search
- Simple vector search (brute-force NumPy or `sqlite-vec`) — scale is small
- Local embeddings via `sentence-transformers` (e.g. `BAAI/bge-small-en-v1.5`), no API key needed
- LLM through a small provider abstraction supporting **Anthropic**, **OpenAI**, and **Ollama** (local, keyless)
- Official MCP Python SDK for the MCP server
- Streamlit for the demo UI (with a simple graph visualisation)
- `pytest` for tests

---

## 5. User Requirements & Configuration

The user should need **only**:

1. Python 3.11+
2. **One** of:
   - an Anthropic API key, or
   - an OpenAI API key, or
   - Ollama installed locally (no key, fully offline)
3. (Optional) Claude Desktop, for the MCP part of the demo

Everything else (embedding model download, database creation) must happen automatically.

Configuration via a `.env` file, with a committed `.env.example`. At minimum: provider choice, API key, model name(s), database path, ports. Never commit secrets. Fail with a **clear, friendly error** if the provider is misconfigured.

---

## 6. Demo User Flow (the experience we are building towards)

This is the story the finished project must support. Use it as the north star for every phase.

1. User starts the app locally with one command and opens the UI.
2. User selects or types a container tag (e.g. `khan`).
3. User sends messages in a chat panel:
   - "I love Adidas sneakers"
   - "My Adidas broke after a month"
   - "I'm switching to Puma"
4. A **memory panel** updates live: extracted facts appear; "Loves Adidas" is shown as outdated (e.g. struck through); a **graph** shows the `UPDATES` link.
5. User asks **"What sneakers should I buy?"** and sees **two answers side by side**:
   - Naive RAG (similarity only) → likely recommends Adidas ❌
   - Memory-backed → recommends Puma ✅
6. **Profile view** shows static vs dynamic facts.
7. **Forgetting:** user says "I have an exam tomorrow", then moves a "simulate time" control forward; the memory stops appearing.
8. **Claude Desktop:** in a brand-new chat, user asks "What do you know about my shoe preferences?" — Claude calls the MCP `recall` tool and answers from the same memory store.
9. **Evaluation view** shows the results table comparing RAG vs memory.

---

## 7. Implementation Phases (in order)

Each phase lists **what** to build and **how we know it's done**. The **how** is up to you.

### Phase 0 — Project setup & installation
- Repo structure, dependency management, `.env.example`, config loading.
- One-command install and one-command run (both documented).
- Health-check endpoint.
- LLM provider abstraction with Anthropic / OpenAI / Ollama, plus a quick connectivity check command.
- **Done when:** a fresh clone can be installed and started following the README, and the provider check passes for at least one provider.

### Phase 1 — Storage, ingestion & RAG baseline
- Database schema for documents, chunks, memories, and memory relationships (edges).
- `add` endpoint: store raw content under a container tag, chunk it, embed chunks.
- Basic semantic search over chunks.
- This naive similarity search is also the **RAG baseline** used for comparison later — keep it accessible as its own mode.
- **Done when:** content can be added and retrieved by semantic search; container tags isolate data; tests cover it.

### Phase 2 — Memory extraction
- On `add`, use the LLM to extract atomic facts with: text, kind (`static`/`dynamic`), and optional expiry.
- Store memories with embeddings and a link to their source document.
- Robust structured output handling (validate, retry or skip gracefully on malformed output).
- **Done when:** adding a multi-fact message produces sensible separate memories; malformed LLM output doesn't crash ingestion.

### Phase 3 — Relationship linking (the heart of the project)
- For each new fact: shortlist similar **current** memories in the same container, then have the LLM classify `NEW` / `UPDATES` / `EXTENDS` / `DUPLICATE`.
- Apply effects as defined in §3.3, including `is_latest` handling and edges.
- Log each decision (for debugging and for showing in the UI).
- **Done when:** the sneaker sequence leaves "switching to Puma" as current and "loves Adidas" as outdated with an `UPDATES` edge; an extending fact creates an `EXTENDS` edge; repeating a fact is skipped. Covered by tests (LLM calls may be mocked in unit tests, plus one optional live integration test).

### Phase 4 — Retrieval: hybrid search, forgetting & profile
- Keyword search + semantic search, fused by rank.
- Filter to current, unexpired memories; support a simulated time offset.
- Search modes: `memories`, `documents` (RAG baseline), `hybrid`.
- `profile` endpoint (static, dynamic, optional query results).
- API shape: `add`, `search`, `profile` — exact signatures are your choice.
- **Done when:** outdated and expired memories never appear in memory search; the profile splits correctly; the API docs page lists all endpoints.

### Phase 5 — Chat with side-by-side comparison
- A chat endpoint that answers a user question twice:
  - **RAG mode:** context from naive chunk similarity only.
  - **Memory mode:** context from profile + memory search.
- Return both answers plus the context each used (so the UI can show *why*).
- Optionally ingest the user's chat messages as new content automatically.
- **Done when:** "What sneakers should I buy?" after the sneaker sequence yields Adidas-leaning RAG context vs Puma in memory mode.

### Phase 6 — MCP server
- Expose three tools:
  - `memory` — save (and optionally forget) information
  - `recall` — search memories, returning results + profile summary
  - `context` — return the full profile for injection at conversation start
- Uses the same engine and database as the API.
- Document the exact Claude Desktop config snippet in the README.
- **Done when:** Claude Desktop can save a fact in one chat and recall it in a new chat.

### Phase 7 — Demo UI
- Supports the full user flow in §6: container tag selector, chat panel, side-by-side answers, live memory list (current vs outdated), relationship graph (colour by relation type), profile view, simulate-time control, evaluation results view.
- A "load sample data" and "reset container" button for quick demos.
- **Done when:** the entire §6 flow (steps 1–7, 9) can be performed in the UI without touching the terminal.

### Phase 8 — Evaluation
- A small scripted dataset (roughly 20–30 scenarios) covering at least:
  - knowledge updates (fact changes, then a question)
  - extensions (detail accumulated over several messages)
  - expiry (temporary fact, then a question after simulated time passes)
- Run each scenario through **RAG baseline** and **memory mode**.
- Report at least **Recall@k** (is the correct current fact in the top-k?) and a **stale-fact rate** (how often an outdated/expired fact appears in the top-k), broken down by category.
- One command to run; outputs a results table (and saved file the UI can display).
- **Done when:** the evaluation runs reproducibly and the results are shown in the README and UI. Report real numbers honestly, even if unflattering, and note failure cases.

### Phase 9 — Documentation & polish
- README: what it is, the Memory-vs-RAG idea, architecture diagram, setup, run, demo walkthrough, API overview, MCP setup, evaluation results, limitations.
- Clean error messages, consistent logging, tests passing.
- Placeholder in README for a demo GIF/screenshots.
- **Done when:** someone unfamiliar with the project can clone, configure one provider, and complete the demo flow using only the README.

### Stretch (only after all phases are complete, then ask me)
- `DERIVES` relation (inferred facts from patterns).
- Recency weighting in ranking.
- Docker / docker-compose.
- Hosted demo (later; not part of this brief).

---

## 8. Out of Scope

Do **not** build: external connectors (Drive, Gmail, Notion, etc.), file uploads / OCR / video / code parsing, authentication or multi-tenant accounts, horizontal scaling, or a hosted deployment.

---

## 9. Working Agreements

- Work **one phase at a time**; summarise what was done and how acceptance criteria were verified at the end of each phase.
- Commit after each phase with a clear message.
- Prefer simple, readable code over clever abstractions. Keep dependencies minimal.
- Keep the engine independent of the API, MCP, and UI layers.
- Write tests for core logic; mock LLM calls in unit tests so the suite runs without an API key.
- If something in this brief is ambiguous or a design choice has a meaningful trade-off, make a sensible decision, note it in the README's "Design decisions" section, and continue. Ask only if a decision would significantly change the scope.
- Start with **Phase 0** now, including installation instructions.