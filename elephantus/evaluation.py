"""Evaluation: naive RAG baseline vs. memory retrieval on scripted scenarios.

    elephantus eval                # uses the configured LLM + embeddings
    elephantus eval --offline      # heuristic stand-in for the LLM + hash embeddings (no key, no download)

For every scenario a fresh container gets the shared filler messages and the
scenario messages (one per simulated hour), then the question is asked at
``ask_at_hours``. Each retrieval mode returns its top-k items and we measure:

* Recall@k        — fraction of the expected (current) facts present in the top-k
* Stale-fact rate — fraction of scenarios whose top-k contains an outdated/expired fact
                    (an item that mentions a stale keyword and none of the expected ones)

Results are written to ``evaluation/results/<label>.json`` and ``.md``.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .config import PROJECT_ROOT, Settings, load_settings
from .embeddings import HashEmbedder, create_embedder
from .engine import MemoryEngine

log = logging.getLogger(__name__)

DATASET = PROJECT_ROOT / "evaluation" / "dataset.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
MODES = {"rag": "documents", "memory": "memories", "hybrid": "hybrid"}
CATEGORIES = ("knowledge_update", "extension", "expiry")


# --------------------------------------------------------------------- scoring
def _mentions(text: str, keywords: list[str]) -> bool:
    text = text.lower()
    return any(re.search(r"(?<![a-z0-9])" + re.escape(k.lower()), text) for k in keywords)


def score(items: list[str], expected: list[list[str]], stale: list[str]) -> dict:
    found = [any(_mentions(t, alts) for t in items) for alts in expected]
    all_expected = [k for alts in expected for k in alts]
    stale_items = [t for t in items if _mentions(t, stale) and not _mentions(t, all_expected)] if stale else []
    return {
        "recall": (sum(found) / len(found)) if expected else None,
        "stale": bool(stale_items),
        "stale_items": stale_items,
    }


def summarise(rows: list[dict]) -> dict:
    """Aggregate per category (and overall) for each mode."""
    summary: dict = {}
    for cat in (*CATEGORIES, "overall"):
        subset = [r for r in rows if cat == "overall" or r["category"] == cat]
        if not subset:
            continue
        summary[cat] = {"n": len(subset)}
        for mode in MODES:
            recalls = [r[mode]["recall"] for r in subset if r[mode]["recall"] is not None]
            stale_applicable = [r for r in subset if r["stale_keywords"]]
            summary[cat][mode] = {
                "recall_at_k": round(sum(recalls) / len(recalls), 3) if recalls else None,
                "stale_rate": (round(sum(r[mode]["stale"] for r in stale_applicable) / len(stale_applicable), 3)
                               if stale_applicable else None),
            }
    return summary


# ------------------------------------------------------------------- running
def run_scenario(engine: MemoryEngine, sc: dict, filler: list[str], k: int) -> dict:
    tag = f"eval-{sc['id']}"
    engine.reset_container(tag)
    errors = []
    for text in filler:
        r = engine.add(text, tag, {"source": "eval-filler"}, time_offset_hours=0)
        errors += [r["error"]] if r.get("error") else []
    for i, text in enumerate(sc["messages"], start=1):
        r = engine.add(text, tag, {"source": "eval"}, time_offset_hours=i)
        errors += [r["error"]] if r.get("error") else []
    row = {"id": sc["id"], "category": sc["category"], "question": sc["question"],
           "stale_keywords": sc["stale"], "errors": errors}
    for name, mode in MODES.items():
        hits = engine.search(sc["question"], tag, mode, k, time_offset_hours=sc["ask_at_hours"])
        texts = [h["text"] for h in hits]
        row[name] = {**score(texts, sc["expected"], sc["stale"]), "top_k": texts}
    row["memories"] = [{"text": m["text"], "status": m["status"]}
                       for m in engine.list_memories(tag, time_offset_hours=sc["ask_at_hours"])]
    engine.reset_container(tag)
    return row


def run(settings: Settings | None = None, *, offline: bool = False, k: int = 3, limit: int | None = None,
        category: str | None = None, workers: int = 4, label: str | None = None) -> dict:
    settings = settings or load_settings()
    data = json.loads(DATASET.read_text())
    scenarios = [s for s in data["scenarios"] if not category or s["category"] == category][:limit]

    if offline:
        llm, embedder = HeuristicLLM(), HashEmbedder()
        provider = "offline heuristic stand-in (NOT an LLM)"
        model, embedding = "rules", "hash"
    else:
        llm, embedder = None, create_embedder(settings)
        provider, model = settings.llm_provider, settings.model
        embedding = settings.embedding_model if settings.embedding_backend == "fastembed" else "hash"

    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        engine = MemoryEngine(settings, llm=llm, embedder=embedder, db_path=Path(tmp) / "eval.db")
        engine.llm  # fail fast with a friendly error if the provider is misconfigured
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            rows = list(pool.map(lambda s: run_scenario(engine, s, data["filler"], k), scenarios))
        engine.close()

    result = {
        "label": label or ("offline-heuristic" if offline else "latest"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": provider, "model": model, "embedding": embedding, "k": k,
        "n_scenarios": len(rows), "seconds": round(time.time() - started, 1),
        "summary": summarise(rows), "scenarios": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{result['label']}.json").write_text(json.dumps(result, indent=1))
    (RESULTS_DIR / f"{result['label']}.md").write_text(to_markdown(result))
    return result


# ------------------------------------------------------------------ reporting
def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.0f}%"


def to_markdown(result: dict) -> str:
    k = result["k"]
    lines = [
        f"**Provider:** {result['provider']} · **model:** {result['model']} · **embeddings:** {result['embedding']}"
        f" · **k:** {k} · **scenarios:** {result['n_scenarios']} · {result['created_at']}",
        "",
        f"| Category | n | RAG Recall@{k} | Memory Recall@{k} | Hybrid Recall@{k} "
        "| RAG stale rate | Memory stale rate | Hybrid stale rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cat, s in result["summary"].items():
        name = f"**{cat}**" if cat == "overall" else cat
        lines.append(f"| {name} | {s['n']} | " + " | ".join(
            [_pct(s[m]["recall_at_k"]) for m in MODES] + [_pct(s[m]["stale_rate"]) for m in MODES]) + " |")
    failures = [r for r in result["scenarios"] if (r["memory"]["recall"] not in (None, 1.0)) or r["memory"]["stale"]]
    lines += ["", f"Memory-mode failures ({len(failures)}):"]
    for r in failures:
        lines.append(f"- `{r['id']}` recall={_pct(r['memory']['recall'])} stale={r['memory']['stale']}"
                     f" top-{k}: {r['memory']['top_k']}")
    return "\n".join(lines) + "\n"


def load_results() -> list[dict]:
    """All saved result files, newest first (used by the UI)."""
    out = []
    for path in RESULTS_DIR.glob("*.json"):
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            log.warning("Could not read %s", path)
    return sorted(out, key=lambda r: r.get("created_at", ""), reverse=True)


# ------------------------------------------------------- offline stand-in LLM
class HeuristicLLM:
    """Rule-based stand-in for the LLM so the evaluation pipeline can run without a provider.

    It is deliberately simple (sentence splitting, keyword cues, word overlap) and is
    NOT representative of LLM quality — use it to sanity-check the pipeline, and run
    `elephantus eval` with a real provider for meaningful numbers.
    """

    name = "heuristic"
    model = "rules"
    _CHANGE = re.compile(r"\b(switch\w*|now|moved|no longer|instead|quit|stopped|gave up|replaced|left|new|"
                         r"cancel\w*|sold|these days|went|can now)\b", re.I)
    _DYNAMIC = re.compile(r"\b(now|currently|this week|this weekend|today|tonight|tomorrow|this afternoon|"
                          r"switching|planning|building|renovating|visiting|until|these days|just)\b", re.I)
    _EXPIRY = [(r"\b(this afternoon|tonight|today)\b", 24), (r"\btomorrow\b", 48),
               (r"\bthis weekend\b", 96), (r"\b(this week|until \w+day)\b", 168)]
    _STOP = frozenset("user i i'm im my me a an the to of and in on at for is am are was with it that this "
                      "these those be been from by as so do does s".split())

    def _words(self, text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9$]+", text.lower()) if w not in self._STOP and len(w) > 1}

    def _to_fact(self, sentence: str) -> str:
        s = sentence.strip().rstrip(".!")
        for pat, rep in [(r"^I'm\b", "User is"), (r"^I am\b", "User is"), (r"^I've\b", "User has"),
                         (r"^I\b", "User"), (r"^My\b", "User's"), (r"\bI'm\b", "user is"),
                         (r"\bI\b", "user"), (r"\bmy\b", "user's"), (r"\bme\b", "user")]:
            s = re.sub(pat, rep, s)
        return s if s.startswith("User") else f"User: {s}"

    def _extract(self, message: str) -> dict:
        out = []
        for sentence in re.split(r"(?<=[.!?])\s+|,\s+(?=I\b|my\b)", message):
            if len(sentence.strip()) < 3 or sentence.strip().endswith("?"):
                continue
            hours = next((h for pat, h in self._EXPIRY if re.search(pat, sentence, re.I)), None)
            kind = "dynamic" if hours or self._DYNAMIC.search(sentence) else "static"
            out.append({"text": self._to_fact(sentence), "kind": kind, "expires_in_hours": hours})
        return {"memories": out}

    def _judge(self, prompt: str) -> dict:
        fact = re.search(r'NEW FACT: "(.*)"', prompt).group(1)
        new = self._words(fact)
        best, best_overlap, best_jac = None, 0, 0.0
        for alias, text in re.findall(r'- id=(m\d+): "(.*)"', prompt):
            old = self._words(text)
            overlap = len(new & old)
            jac = overlap / max(1, len(new | old))
            if overlap > best_overlap:
                best, best_overlap, best_jac = alias, overlap, jac
        if best is None:
            return {"relation": "NEW", "target_ids": [], "reason": "no overlap"}
        if best_jac >= 0.8:
            return {"relation": "DUPLICATE", "target_ids": [best], "reason": "near-identical wording"}
        if self._CHANGE.search(fact):
            return {"relation": "UPDATES", "target_ids": [best], "reason": "change cue + shared topic"}
        return {"relation": "EXTENDS", "target_ids": [best], "reason": "shared topic"}

    def complete(self, system, messages, max_tokens=2000, json_mode=False, effort="low"):
        prompt = messages[-1]["content"]
        if system.startswith("You extract"):
            return json.dumps(self._extract(prompt.split('"""')[1].strip()))
        if system.startswith("You maintain a memory"):
            return json.dumps(self._judge(prompt))
        return "(offline heuristic mode cannot write answers)"
