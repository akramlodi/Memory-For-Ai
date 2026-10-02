"""Command line entry point: `mini-sm <command>`."""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_settings, setup_logging


def cmd_check(args) -> int:
    from .llm import LLMError, check_llm, create_llm

    settings = load_settings()
    print(f"Provider : {settings.llm_provider}")
    print(f"Model    : {settings.model}")
    try:
        llm = create_llm(settings)
        reply = check_llm(llm)
    except (ConfigError, LLMError) as exc:
        print(f"\n[FAIL] {exc}")
        return 1
    print(f"Reply    : {reply!r}")
    print("\n[OK] LLM provider is reachable.")
    return 0


def cmd_api(args) -> int:
    import uvicorn

    settings = load_settings()
    uvicorn.run("mini_supermemory.api:app", host=settings.api_host, port=settings.api_port)
    return 0


def cmd_start(args) -> int:
    """Run the REST API, which also serves the web UI at /."""
    import uvicorn

    settings = load_settings()
    print(f"Web UI   : http://{settings.api_host}:{settings.api_port}/")
    print(f"REST API : http://{settings.api_host}:{settings.api_port}/docs")
    if not args.no_browser:
        import threading
        import webbrowser

        threading.Timer(1.5, webbrowser.open, [f"http://{settings.api_host}:{settings.api_port}/"]).start()
    uvicorn.run("mini_supermemory.api:app", host=settings.api_host, port=settings.api_port)
    return 0


def cmd_mcp(args) -> int:
    from .mcp_server import main as mcp_main

    mcp_main()
    return 0


def cmd_eval(args) -> int:
    from .evaluation import run, to_markdown
    from .llm import LLMError

    try:
        result = run(offline=args.offline, k=args.k, limit=args.limit, category=args.category,
                     workers=args.workers, label=args.label)
    except LLMError as exc:
        print(f"[FAIL] {exc}")
        return 1
    print(to_markdown(result))
    print(f"Saved to evaluation/results/{result['label']}.json and .md ({result['seconds']}s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mini-sm", description="Mini-Supermemory command line")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="check that the configured LLM provider works").set_defaults(func=cmd_check)
    for name in ("start", "ui"):
        sp = sub.add_parser(name, help="run the web UI + REST API (opens the browser)")
        sp.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
        sp.set_defaults(func=cmd_start)
    sub.add_parser("api", help="run the REST API (also serves the web UI at /)").set_defaults(func=cmd_api)
    sub.add_parser("mcp", help="run the MCP server over stdio (for Claude Desktop)").set_defaults(func=cmd_mcp)
    ev = sub.add_parser("eval", help="run the RAG-vs-memory evaluation")
    ev.add_argument("--offline", action="store_true",
                    help="use a rule-based stand-in for the LLM and hash embeddings (pipeline sanity check only)")
    ev.add_argument("--k", type=int, default=3, help="top-k for Recall@k (default 3)")
    ev.add_argument("--limit", type=int, default=None, help="only run the first N scenarios")
    ev.add_argument("--category", choices=["knowledge_update", "extension", "expiry"], default=None)
    ev.add_argument("--workers", type=int, default=4, help="scenarios run in parallel (default 4)")
    ev.add_argument("--label", default=None, help="results file name (default: latest / offline-heuristic)")
    ev.set_defaults(func=cmd_eval)
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        setup_logging(settings.log_level)
        return args.func(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
