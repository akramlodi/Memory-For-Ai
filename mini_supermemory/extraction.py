"""Memory extraction: turn a raw message into short, atomic facts using the LLM.

Robustness: LLM output is parsed leniently (code fences / chatter around the JSON
are tolerated), validated item by item, retried once with a corrective message,
and if it is still unusable the message simply yields no memories — ingestion
never crashes because of a bad LLM reply.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from .llm import LLM, LLMError

log = logging.getLogger(__name__)


class ExtractedFact(BaseModel):
    text: str = Field(min_length=3, max_length=400)
    kind: Literal["static", "dynamic"] = "static"
    expires_in_hours: float | None = None

    @field_validator("kind", mode="before")
    @classmethod
    def _lower(cls, v):
        return str(v).strip().lower() if v is not None else "static"

    @field_validator("expires_in_hours", mode="before")
    @classmethod
    def _positive(cls, v):
        if v in (None, "", "null", 0):
            return None
        v = float(v)
        return v if v > 0 else None


EXTRACT_SYSTEM = """You extract long-term memory facts about a user from a message they wrote.

Rules:
- Return ONLY facts the message states or clearly implies about the user (preferences, plans,
  events, relationships, possessions, opinions, circumstances). Ignore questions, greetings and
  requests for help: they contain no facts, so return an empty list for them.
- Each fact must be atomic (one idea), self-contained, written in the third person starting
  with "User" (e.g. "User loves Adidas sneakers"), and keep concrete names/brands/numbers.
- Resolve pronouns and vague references using the known facts provided, if any.
- kind: "static" for stable, long-term facts (identity, where they study/work, lasting
  preferences); "dynamic" for recent or ongoing context (current projects, plans, what they
  are doing now, recent events).
- expires_in_hours: ONLY for facts that are clearly time-bound and will stop being relevant
  (e.g. "I have an exam tomorrow" -> about 48; "meeting this afternoon" -> about 12;
  "I'm in Paris this week" -> about 168). Use null for everything else.

Output format (JSON):
{"memories": [{"text": "...", "kind": "static" | "dynamic", "expires_in_hours": number | null}]}"""


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_object(text: str) -> dict:
    """Parse the first JSON object in `text`, tolerating code fences and surrounding prose."""
    text = _FENCE.sub("", text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON is not an object")
    return obj


def call_json(llm: LLM, system: str, user: str, *, retries: int = 1, max_tokens: int = 2000) -> dict | None:
    """Ask for JSON; on malformed output retry with a corrective message. None if it never parses."""
    messages = [{"role": "user", "content": user}]
    for attempt in range(retries + 1):
        raw = llm.complete(system, messages, max_tokens=max_tokens, json_mode=True)
        try:
            return parse_json_object(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            log.warning("Malformed JSON from LLM (attempt %d): %s | %.200r", attempt + 1, exc, raw)
            messages += [
                {"role": "assistant", "content": raw or "(empty)"},
                {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object."},
            ]
    return None


def _format_time(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%A %Y-%m-%d %H:%M UTC")


def extract_facts(llm: LLM, message: str, now_ts: float, known_facts: list[str] | None = None) -> list[ExtractedFact]:
    known = "\n".join(f"- {f}" for f in (known_facts or [])) or "(none)"
    user = (
        f"Current date/time: {_format_time(now_ts)}\n\n"
        f"Known facts about the user (context only; do not repeat them):\n{known}\n\n"
        f"Message:\n\"\"\"\n{message}\n\"\"\""
    )
    try:
        data = call_json(llm, EXTRACT_SYSTEM, user)
    except LLMError as exc:
        log.error("Extraction failed: %s", exc)
        raise
    if data is None:
        log.warning("Extraction gave up after retries; storing the document without memories.")
        return []
    items = data.get("memories")
    if not isinstance(items, list):
        log.warning("Extraction JSON has no 'memories' list: %.200r", data)
        return []
    facts: list[ExtractedFact] = []
    for item in items:
        if isinstance(item, str):
            item = {"text": item}
        try:
            facts.append(ExtractedFact.model_validate(item))
        except (ValidationError, TypeError, ValueError) as exc:
            log.warning("Skipping invalid extracted fact %.200r: %s", item, exc)
    return facts
