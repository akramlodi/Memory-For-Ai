"""MCP server exposing the memory engine to Claude Desktop (or any MCP client).

Tools:
* ``memory``  — save information, or forget it (action="forget")
* ``recall``  — search memories; returns results plus a profile summary
* ``context`` — the full profile, for injection at the start of a conversation

Runs over stdio:  python -m elephantus.mcp_server   (or: elephantus mcp)
Uses the same engine and SQLite file as the REST API and the UI.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

from mcp.server.mcpserver import MCPServer

from .config import ConfigError, load_settings, setup_logging
from .engine import MemoryEngine

log = logging.getLogger(__name__)

INSTRUCTIONS = """Long-term memory about the user.
- Call `context` at the start of a conversation to load what you know about the user.
- Call `recall` before answering questions about the user's preferences, plans or history.
- Call `memory` to save new facts the user shares (or to forget one when asked).
Memories are kept current: outdated and expired facts are not returned."""


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {t}" for t in items) or "- (none)"


def build_server(get_engine: Callable[[], MemoryEngine], default_tag: str) -> MCPServer:
    server = MCPServer("elephantus", instructions=INSTRUCTIONS)

    @server.tool()
    def memory(content: str, action: Literal["save", "forget"] = "save", container_tag: str | None = None) -> str:
        """Save information about the user to long-term memory, or forget it.

        Args:
            content: What to remember (e.g. "I'm switching to Puma sneakers"), or, with
                action="forget", a description of the memory to forget.
            action: "save" (default) or "forget".
            container_tag: Memory space to use (defaults to the server's configured tag).
        """
        tag = container_tag or default_tag
        engine = get_engine()
        if action == "forget":
            forgotten = engine.forget(tag, content=content)
            return f"Forgot: {forgotten['text']}" if forgotten else "No matching memory found to forget."
        result = engine.add(content, tag, {"source": "mcp"})
        if result.get("error"):
            return f"Saved the message, but extracting memories failed: {result['error']}"
        lines = [f"- [{m['decision']}] {m['fact']}" for m in result["memories"]]
        return "Saved. Extracted memories:\n" + ("\n".join(lines) or "- (no facts found)")

    @server.tool()
    def recall(query: str, container_tag: str | None = None, limit: int = 5) -> str:
        """Search the user's memories (only current, non-expired facts) and summarise their profile.

        Args:
            query: What to look for, e.g. "shoe preferences".
            container_tag: Memory space to search (defaults to the server's configured tag).
            limit: Maximum number of memories to return.
        """
        tag = container_tag or default_tag
        prof = get_engine().profile(tag, q=query, limit=limit)
        results = [f"{r['text']}  (kind={r['kind']})" for r in prof.get("search_results", [])]
        return (f"Relevant memories for '{query}':\n{_bullets(results)}\n\n"
                f"Profile summary:\nStable facts:\n{_bullets(prof['static'])}\n"
                f"Recent context:\n{_bullets(prof['dynamic'])}")

    @server.tool()
    def context(container_tag: str | None = None) -> str:
        """Return everything currently known about the user, for use at the start of a conversation.

        Args:
            container_tag: Memory space to read (defaults to the server's configured tag).
        """
        return get_engine().context_prompt(container_tag or default_tag)

    return server


def main() -> None:
    try:
        settings = load_settings()
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}")
    setup_logging(settings.log_level)  # logs go to stderr; stdout carries the MCP protocol
    engine: MemoryEngine | None = None

    def get_engine() -> MemoryEngine:
        nonlocal engine
        if engine is None:
            engine = MemoryEngine(settings)
        return engine

    log.info("Starting MCP server (db=%s, default tag=%s)", settings.database_path, settings.default_container_tag)
    build_server(get_engine, settings.default_container_tag).run("stdio")


if __name__ == "__main__":
    main()
