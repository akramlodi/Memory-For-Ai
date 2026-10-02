"""Streamlit demo UI — a thin layer over the memory engine (same database as API and MCP).

Run:  mini-sm ui    (or: streamlit run mini_supermemory/ui.py)
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from mini_supermemory.config import ConfigError, load_settings, setup_logging
from mini_supermemory.engine import MemoryEngine
from mini_supermemory.evaluation import MODES, load_results, to_markdown
from mini_supermemory.llm import LLMError
from mini_supermemory.sample_data import SAMPLE_MESSAGES

st.set_page_config(page_title="Mini-Supermemory", page_icon="🧠", layout="wide")

STATUS_STYLE = {
    "current": ("#1a7f37", ""),
    "outdated": ("#8c959f", "text-decoration: line-through;"),
    "expired": ("#bf8700", "text-decoration: line-through;"),
    "forgotten": ("#cf222e", "text-decoration: line-through;"),
}
RELATION_COLOR = {"UPDATES": "#cf222e", "EXTENDS": "#0969da"}
DECISION_ICON = {"NEW": "🆕", "UPDATES": "🔁", "EXTENDS": "➕", "DUPLICATE": "♻️", "FORGET": "🗑️"}


@st.cache_resource(show_spinner="Loading memory engine (first run downloads the embedding model)...")
def get_engine() -> MemoryEngine:
    settings = load_settings()
    setup_logging(settings.log_level)
    return MemoryEngine(settings)


def fmt_time(ts: float | None) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%b %d %H:%M UTC") if ts else "—"


def run_safely(fn, *args, **kwargs):
    """Run an engine call, turning provider/config problems into friendly UI errors."""
    try:
        return fn(*args, **kwargs)
    except (LLMError, ConfigError) as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.warning(str(exc))
    return None


# ------------------------------------------------------------------ sidebar
try:
    engine = get_engine()
except ConfigError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

settings = engine.settings
st.sidebar.title("🧠 Mini-Supermemory")
st.sidebar.caption(f"LLM: **{settings.llm_provider}** · `{settings.model}`  \nEmbeddings: `{settings.embedding_backend}`")

existing = engine.list_containers()
choice = st.sidebar.selectbox("Container tag", options=existing + ["➕ new tag..."],
                              index=existing.index("khan") if "khan" in existing else len(existing))
tag = st.sidebar.text_input("New container tag", value="khan") if choice == "➕ new tag..." else choice
tag = (tag or "").strip()
if not tag:
    st.info("Enter a container tag to begin.")
    st.stop()

offset = st.sidebar.slider("⏩ Simulate time (hours from now)", 0, 336, 0, step=6,
                           help="Pretend this much time has passed: time-bound memories expire.")
if offset:
    st.sidebar.info(f"Simulated now: {fmt_time(engine.now(offset))}")

compare = st.sidebar.toggle("Answer every message (RAG vs Memory)", value=True)
remember = st.sidebar.toggle("Remember my messages", value=True)
k = st.sidebar.number_input("Context items per mode (k)", 1, 10, 3)

if st.sidebar.button("📥 Load sample data", use_container_width=True):
    progress = st.sidebar.progress(0.0, text="Ingesting...")
    for i, msg in enumerate(SAMPLE_MESSAGES, start=1):
        if run_safely(engine.add, msg, tag, {"source": "sample"}, offset) is None:
            break
        progress.progress(i / len(SAMPLE_MESSAGES), text=msg)
    st.rerun()
if st.sidebar.button("🗑️ Reset container", use_container_width=True):
    engine.reset_container(tag)
    st.session_state.pop(f"history:{tag}", None)
    st.rerun()

history: list[dict] = st.session_state.setdefault(f"history:{tag}", [])


# ------------------------------------------------------------- components
def memory_badge(m: dict) -> str:
    color, extra = STATUS_STYLE[m["status"]]
    expiry = f" · expires {fmt_time(m['expires_at'])}" if m["expires_at"] else ""
    return (f"<div style='margin:2px 0;padding:4px 8px;border-left:4px solid {color};{extra}'>"
            f"{html.escape(m['text'])}<br><span style='font-size:0.75em;color:{color}'>"
            f"{m['status']} · {m['kind']}{expiry}</span></div>")


def memory_panel(memories: list[dict]) -> None:
    current = [m for m in memories if m["status"] == "current"]
    old = [m for m in memories if m["status"] != "current"]
    st.markdown(f"**Current memories ({len(current)})**")
    st.markdown("".join(memory_badge(m) for m in current) or "_none yet_", unsafe_allow_html=True)
    if old:
        st.markdown(f"**Outdated / expired / forgotten ({len(old)})**")
        st.markdown("".join(memory_badge(m) for m in old), unsafe_allow_html=True)


def graph_dot(graph: dict) -> str:
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];']
    for n in graph["nodes"]:
        color = STATUS_STYLE[n["status"]][0]
        fill = "#dafbe1" if n["status"] == "current" else "#f6f8fa"
        label = esc(n["text"] if len(n["text"]) < 60 else n["text"][:57] + "...")
        lines.append(f'"{n["id"]}" [label="{label}\\n({n["status"]}, {n["kind"]})", color="{color}", '
                     f'fillcolor="{fill}", fontcolor="{"#24292f" if n["status"] == "current" else "#8c959f"}"];')
    for e in graph["edges"]:
        c = RELATION_COLOR.get(e["relation"], "black")
        lines.append(f'"{e["source_id"]}" -> "{e["target_id"]}" [label="{e["relation"]}", color="{c}", fontcolor="{c}"];')
    lines.append("}")
    return "\n".join(lines)


def show_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
        for d in turn.get("decisions", []):
            st.caption(f"{DECISION_ICON.get(d['decision'], '')} **{d['decision']}** — {d['fact']}")
        if turn.get("error"):
            st.warning(turn["error"])
    if turn.get("rag"):
        with st.chat_message("assistant"):
            left, right = st.columns(2)
            with left:
                st.markdown("##### 📄 Naive RAG")
                st.write(turn["rag"]["answer"])
                with st.expander("Context used (similar chunks)"):
                    st.text(turn["rag"]["prompt_context"])
            with right:
                st.markdown("##### 🧠 Memory")
                st.write(turn["memory"]["answer"])
                with st.expander("Context used (profile + memories)"):
                    st.text(turn["memory"]["prompt_context"])


# --------------------------------------------------------------------- tabs
tab_chat, tab_graph, tab_profile, tab_eval, tab_log = st.tabs(
    ["💬 Chat", "🕸️ Memory graph", "👤 Profile", "📊 Evaluation", "📝 Decision log"])

with tab_chat:
    chat_col, mem_col = st.columns([3, 2])
    with chat_col:
        for turn in history:
            show_turn(turn)
        prompt = st.chat_input(f"Message as '{tag}' (e.g. I love Adidas sneakers)")
        if prompt:
            turn = {"question": prompt}
            with st.spinner("Thinking..."):
                if compare:
                    out = run_safely(engine.chat, prompt, tag, int(k), offset, remember)
                    if out:
                        turn.update(rag=out["rag"], memory=out["memory"])
                        ingested = out.get("ingested") or {}
                        turn["decisions"], turn["error"] = ingested.get("memories", []), ingested.get("error")
                elif remember:
                    out = run_safely(engine.add, prompt, tag, {"source": "chat"}, offset)
                    if out:
                        turn["decisions"], turn["error"] = out["memories"], out.get("error")
            history.append(turn)
            st.rerun()
    with mem_col:
        memories = engine.list_memories(tag, True, offset)
        memory_panel(memories)
        g = engine.graph(tag, offset)
        if g["edges"]:
            st.graphviz_chart(graph_dot(g), use_container_width=True)

with tab_graph:
    g = engine.graph(tag, offset)
    st.caption("Edges point from the newer memory to the one it relates to. "
               "🔴 UPDATES (old fact no longer current) · 🔵 EXTENDS (adds detail, both current)")
    if g["nodes"]:
        st.graphviz_chart(graph_dot(g), use_container_width=True)
    else:
        st.info("No memories yet. Send a message or load the sample data.")

with tab_profile:
    prof = engine.profile(tag, time_offset_hours=offset)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Static (long-term)")
        st.markdown("\n".join(f"- {t}" for t in prof["static"]) or "_none_")
    with c2:
        st.subheader("Dynamic (recent)")
        st.markdown("\n".join(f"- {t}" for t in prof["dynamic"]) or "_none_")
    st.divider()
    q = st.text_input("Search memories", placeholder="What sneakers should I buy?")
    mode = st.radio("Mode", ["memories", "documents", "hybrid"], horizontal=True,
                    help="memories = current facts (hybrid keyword+semantic); documents = naive RAG baseline")
    if q:
        hits = run_safely(engine.search, q, tag, mode, int(k) + 2, offset) or []
        st.dataframe(pd.DataFrame([{"type": h["type"], "text": h["text"], "score": h["score"]} for h in hits]),
                     use_container_width=True, hide_index=True)

with tab_eval:
    results = load_results()
    if not results:
        st.info("No evaluation results yet. Run `mini-sm eval` (or `mini-sm eval --offline`).")
    else:
        labels = [f"{r['label']} — {r['provider']} / {r['model']} ({r['created_at']})" for r in results]
        r = results[st.selectbox("Results file", range(len(results)), format_func=lambda i: labels[i])]
        if "heuristic" in r["provider"]:
            st.warning("These numbers come from the offline rule-based stand-in, not an LLM. "
                       "They only sanity-check the pipeline; run `mini-sm eval` with a provider for real numbers.")
        st.markdown(to_markdown(r).split("\nMemory-mode failures")[0])
        rows = []
        for cat, s in r["summary"].items():
            for m in MODES:
                rows.append({"category": cat, "mode": m, "Recall@k": s[m]["recall_at_k"], "stale rate": s[m]["stale_rate"]})
        df = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        c1.markdown(f"**Recall@{r['k']}** (higher is better)")
        c1.bar_chart(df.pivot(index="category", columns="mode", values="Recall@k"), stack=False)
        c2.markdown("**Stale-fact rate** (lower is better)")
        c2.bar_chart(df.pivot(index="category", columns="mode", values="stale rate"), stack=False)
        with st.expander("Per-scenario details"):
            st.dataframe(pd.DataFrame([{
                "id": s["id"], "category": s["category"], "question": s["question"],
                "RAG recall": s["rag"]["recall"], "Memory recall": s["memory"]["recall"],
                "RAG stale": s["rag"]["stale"], "Memory stale": s["memory"]["stale"],
                "Memory top-k": " | ".join(s["memory"]["top_k"]), "RAG top-k": " | ".join(s["rag"]["top_k"]),
            } for s in r["scenarios"]]), use_container_width=True, hide_index=True)

with tab_log:
    log_rows = engine.link_log(tag)
    if log_rows:
        st.dataframe(pd.DataFrame([{
            "time": fmt_time(e["created_at"]), "decision": f"{DECISION_ICON.get(e['decision'], '')} {e['decision']}",
            "fact": e["fact"], "reason": e["reason"]} for e in log_rows]), use_container_width=True, hide_index=True)
    else:
        st.info("No linking decisions yet.")
    with st.expander("Raw documents"):
        for d in engine.list_documents(tag):
            st.markdown(f"- `{fmt_time(d['created_at'])}` {d['content']}")
