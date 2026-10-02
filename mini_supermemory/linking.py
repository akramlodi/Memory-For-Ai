"""Relationship linking: decide how a new fact relates to existing current memories.

Two steps:
1. The engine shortlists the most similar *current* memories in the same container
   using embeddings (cheap).
2. The LLM judges the relation against that shortlist (this module).

Labels: NEW | UPDATES | EXTENDS | DUPLICATE (see README for the semantics).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .extraction import call_json
from .llm import LLM

log = logging.getLogger(__name__)

RELATIONS = ("NEW", "UPDATES", "EXTENDS", "DUPLICATE")

JUDGE_SYSTEM = """You maintain a memory of facts about a user. A NEW FACT has just been learned.
Compare it with the EXISTING FACTS and classify the relationship:

- "UPDATES": the new fact replaces or contradicts existing fact(s), so they are no longer
  currently true (a changed preference, a changed state, a replaced value, a decision that
  supersedes an old one). Example: "User loves Adidas sneakers" is updated by
  "User is switching to Puma sneakers".
- "EXTENDS": the new fact adds detail to existing fact(s) and they all remain true.
  Example: "User works at Acme" is extended by "User is a backend engineer at Acme".
- "DUPLICATE": the new fact says the same thing as an existing fact (no new information).
- "NEW": the new fact is unrelated to all existing facts.

"target_ids" lists the ids of the existing facts that are updated / extended / duplicated
(empty for NEW). Only use ids from the list.

Output format (JSON):
{"relation": "NEW" | "UPDATES" | "EXTENDS" | "DUPLICATE", "target_ids": ["m1"], "reason": "<one short sentence>"}"""


@dataclass
class Candidate:
    id: str
    text: str
    similarity: float


@dataclass
class Judgement:
    relation: str
    target_ids: list[str] = field(default_factory=list)
    reason: str = ""


def judge_relation(llm: LLM, fact: str, candidates: list[Candidate]) -> Judgement:
    if not candidates:
        return Judgement("NEW", [], "No similar memories in this container.")

    # Short aliases (m1, m2...) are easier for the model to copy correctly than UUIDs.
    alias = {f"m{i}": c.id for i, c in enumerate(candidates, start=1)}
    listing = "\n".join(f'- id={a}: "{c.text}"' for a, c in zip(alias, candidates))
    user = f'EXISTING FACTS:\n{listing}\n\nNEW FACT: "{fact}"'

    data = call_json(llm, JUDGE_SYSTEM, user, max_tokens=1000)
    if data is None:
        return Judgement("NEW", [], "Relation judge returned invalid output; stored as new.")

    relation = str(data.get("relation", "")).strip().upper()
    if relation not in RELATIONS:
        log.warning("Unknown relation %r; treating as NEW", relation)
        return Judgement("NEW", [], f"Unknown relation {relation!r}; stored as new.")

    raw_targets = data.get("target_ids") or []
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    real_ids = set(alias.values())
    targets: list[str] = []
    for t in raw_targets:
        t = str(t).strip()
        real = alias.get(t) or (t if t in real_ids else None)
        if real and real not in targets:
            targets.append(real)

    reason = str(data.get("reason", ""))[:300]
    if relation in ("UPDATES", "EXTENDS") and not targets:
        # A relation without a valid target cannot be applied; keep the fact rather than lose it.
        return Judgement("NEW", [], f"{relation} without a valid target; stored as new. {reason}".strip())
    if relation == "NEW":
        targets = []
    return Judgement(relation, targets, reason)
