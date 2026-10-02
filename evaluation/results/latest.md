**Provider:** azure · **model:** gpt-4o · **embeddings:** BAAI/bge-small-en-v1.5 · **k:** 3 · **scenarios:** 25 · 2026-10-02T04:52:59+00:00

| Category | n | RAG Recall@3 | Memory Recall@3 | Hybrid Recall@3 | RAG stale rate | Memory stale rate | Hybrid stale rate |
|---|---|---|---|---|---|---|---|
| knowledge_update | 11 | 100% | 100% | 91% | 100% | 27% | 82% |
| extension | 7 | 93% | 96% | 58% | n/a | n/a | n/a |
| expiry | 7 | 100% | 100% | 100% | 100% | 0% | 43% |
| **overall** | 25 | 98% | 99% | 84% | 100% | 17% | 67% |

Memory-mode failures (4):
- `upd-sneakers` recall=100% stale=True top-3: ["User's Adidas sneakers broke after a month.", 'User is switching to Puma.', 'User is learning Spanish on Duolingo.']
- `upd-manager` recall=100% stale=True top-3: ["User's new manager is David.", "User's previous manager, Sarah, left the company.", 'User has a golden retriever named Max.']
- `upd-exercise` recall=100% stale=True top-3: ['User cancelled their gym membership.', 'User is learning Spanish on Duolingo.', 'User swims at the community pool.']
- `ext-work` recall=75% stale=False top-3: ['User is a backend engineer.', 'User works at Acme Corp', 'User mostly writes Rust.']
