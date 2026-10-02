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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mini-sm", description="Mini-Supermemory command line")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="check that the configured LLM provider works").set_defaults(func=cmd_check)
    sub.add_parser("api", help="run the REST API").set_defaults(func=cmd_api)
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
