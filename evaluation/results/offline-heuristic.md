**Provider:** offline heuristic stand-in (NOT an LLM) · **model:** rules · **embeddings:** hash · **k:** 3 · **scenarios:** 25 · 2026-10-02T04:03:49+00:00

| Category | n | RAG Recall@3 | Memory Recall@3 | Hybrid Recall@3 | RAG stale rate | Memory stale rate | Hybrid stale rate |
|---|---|---|---|---|---|---|---|
| knowledge_update | 11 | 73% | 73% | 55% | 91% | 55% | 91% |
| extension | 7 | 44% | 49% | 37% | n/a | n/a | n/a |
| expiry | 7 | 83% | 83% | 67% | 57% | 0% | 29% |
| **overall** | 25 | 67% | 68% | 52% | 78% | 33% | 67% |

Memory-mode failures (14):
- `upd-sneakers` recall=0% stale=True top-3: ['User love Adidas sneakers', 'User have a golden retriever named Max', "User's Adidas broke after a month"]
- `upd-city` recall=100% stale=True top-3: ['User live in Chicago', 'User just moved to Seattle for a new job', 'User have a golden retriever named Max']
- `upd-phone` recall=100% stale=True top-3: ["User replaced user's old phone with a Google Pixel 8", 'User use an iPhone 12', 'User have a golden retriever named Max']
- `upd-editor` recall=0% stale=False top-3: ['User have a golden retriever named Max', 'User is learning Spanish on Duolingo', 'User play the guitar on weekends']
- `upd-commute` recall=100% stale=True top-3: ['User drive a Honda Civic to work', "User's favourite cuisine is Thai food", "User sold user's car and now commute by bike"]
- `upd-coffee` recall=0% stale=True top-3: ['User drink three cups of coffee a day', 'User is learning Spanish on Duolingo', 'User have a golden retriever named Max']
- `upd-exercise` recall=100% stale=True top-3: ["User cancelled user's gym membership", "User's favourite cuisine is Thai food", 'User swim at the community pool now']
- `ext-work` recall=0% stale=False top-3: ["User's favourite cuisine is Thai food", 'User play the guitar on weekends', 'User is learning Spanish on Duolingo']
- `ext-trip` recall=67% stale=False top-3: ['User is planning a trip to Japan', 'User: The Japan trip is in April', "User's favourite cuisine is Thai food"]
- `ext-allergy` recall=0% stale=False top-3: ['User is learning Spanish on Duolingo', "User's favourite cuisine is Thai food", 'User have a golden retriever named Max']
- `ext-sister` recall=75% stale=False top-3: ['User have a sister named Priya', "User's favourite cuisine is Thai food", 'User: Priya is a doctor in Toronto']
- `ext-app` recall=33% stale=False top-3: ['User is building it with Flutter', 'User is building a mobile app', 'User have a golden retriever named Max']
- `ext-music` recall=67% stale=False top-3: ['User collect jazz records on vinyl', 'User play the guitar on weekends', 'User have a golden retriever named Max']
- `exp-tokyo` recall=0% stale=False top-3: ['User is learning Spanish on Duolingo', 'User have a golden retriever named Max', 'User play the guitar on weekends']
