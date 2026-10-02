"""A scripted stand-in for the LLM so the test-suite runs without any provider."""

from __future__ import annotations

import json
import re

_LISTING = re.compile(r'- id=(m\d+): "(.*)"')


class FakeLLM:
    name = "fake"
    model = "fake-1"

    def __init__(self, facts=None, relations=None, raw_extraction=None, answer=None):
        self.facts = facts or {}          # message -> list[dict | str]
        self.relations = relations or {}  # fact text -> (relation, [target texts])
        self.raw_extraction = raw_extraction  # list of raw strings returned in order (to test malformed output)
        self.answer = answer or (lambda system, prompt: "ANSWER: " + prompt[-80:])
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, messages, max_tokens=2000, json_mode=False, effort="low"):
        prompt = messages[-1]["content"]
        self.calls.append((system[:40], prompt))
        if system.startswith("You extract"):
            if self.raw_extraction:
                return self.raw_extraction.pop(0)
            message = prompt.split('"""')[1].strip()
            return json.dumps({"memories": self.facts.get(message, [])})
        if system.startswith("You maintain a memory"):
            fact = re.search(r'NEW FACT: "(.*)"', prompt).group(1)
            aliases = {text: alias for alias, text in _LISTING.findall(prompt)}
            relation, targets = self.relations.get(fact, ("NEW", []))
            return json.dumps({"relation": relation, "target_ids": [aliases[t] for t in targets if t in aliases],
                               "reason": "scripted"})
        return self.answer(system, prompt)
