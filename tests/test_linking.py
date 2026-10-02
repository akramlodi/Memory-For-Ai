import json

import pytest

from elephantus.linking import Candidate, judge_relation
from tests.fakes import FakeLLM

ADIDAS = "User loves Adidas sneakers"
BROKE = "User's Adidas sneakers broke after a month"
PUMA = "User is switching to Puma sneakers"


@pytest.fixture
def sneaker_llm(fake_llm):
    fake_llm.facts.update({
        "I love Adidas sneakers": [{"text": ADIDAS, "kind": "static"}],
        "My Adidas broke after a month": [{"text": BROKE, "kind": "dynamic"}],
        "I'm switching to Puma": [{"text": PUMA, "kind": "dynamic"}],
    })
    fake_llm.relations.update({BROKE: ("EXTENDS", [ADIDAS]), PUMA: ("UPDATES", [ADIDAS])})
    return fake_llm


def test_sneaker_sequence(engine, sneaker_llm):
    for msg in ["I love Adidas sneakers", "My Adidas broke after a month", "I'm switching to Puma"]:
        engine.add(msg, "khan")
    status = {m["text"]: m["status"] for m in engine.list_memories("khan")}
    assert status == {ADIDAS: "outdated", BROKE: "current", PUMA: "current"}

    graph = engine.graph("khan")
    ids = {n["text"]: n["id"] for n in graph["nodes"]}
    edges = {(e["source_id"], e["target_id"], e["relation"]) for e in graph["edges"]}
    assert (ids[PUMA], ids[ADIDAS], "UPDATES") in edges
    assert (ids[BROKE], ids[ADIDAS], "EXTENDS") in edges

    decisions = [d["decision"] for d in reversed(engine.link_log("khan"))]
    assert decisions == ["NEW", "EXTENDS", "UPDATES"]


def test_duplicate_is_skipped(engine, fake_llm):
    fake_llm.facts["I love tea"] = [{"text": "User loves tea"}]
    fake_llm.facts["Tea is my favourite"] = [{"text": "User's favourite drink is tea"}]
    fake_llm.relations["User's favourite drink is tea"] = ("DUPLICATE", ["User loves tea"])
    engine.add("I love tea", "khan")
    result = engine.add("Tea is my favourite", "khan")
    assert result["memories"][0]["decision"] == "DUPLICATE" and result["memories"][0]["memory_id"] is None
    assert [m["text"] for m in engine.list_memories("khan")] == ["User loves tea"]


def test_linking_never_crosses_containers(engine, sneaker_llm):
    engine.add("I love Adidas sneakers", "work")
    result = engine.add("I'm switching to Puma", "khan")
    # No candidates in 'khan', so the judge is not even consulted and nothing in 'work' changes.
    assert result["memories"][0]["decision"] == "NEW"
    assert engine.list_memories("work")[0]["status"] == "current"


def test_expired_memories_are_not_link_candidates(engine, fake_llm):
    fake_llm.facts["exam tomorrow"] = [{"text": "User has an exam tomorrow", "kind": "dynamic", "expires_in_hours": 48}]
    fake_llm.facts["exam later"] = [{"text": "User has an exam next month", "kind": "dynamic"}]
    fake_llm.relations["User has an exam next month"] = ("UPDATES", ["User has an exam tomorrow"])
    engine.add("exam tomorrow", "khan")
    result = engine.add("exam later", "khan", time_offset_hours=72)
    assert result["memories"][0]["decision"] == "NEW"


def test_judge_validation():
    cands = [Candidate("mem_a", "User loves Adidas", 0.9)]
    bad_target = FakeLLM(raw_extraction=None)
    bad_target.complete = lambda *a, **k: json.dumps({"relation": "UPDATES", "target_ids": ["m9"]})
    assert judge_relation(bad_target, "x", cands).relation == "NEW"

    unknown = FakeLLM()
    unknown.complete = lambda *a, **k: '{"relation": "MERGES", "target_ids": ["m1"]}'
    assert judge_relation(unknown, "x", cands).relation == "NEW"

    real_id = FakeLLM()
    real_id.complete = lambda *a, **k: '{"relation": "updates", "target_ids": "mem_a"}'
    j = judge_relation(real_id, "x", cands)
    assert (j.relation, j.target_ids) == ("UPDATES", ["mem_a"])

    garbage = FakeLLM()
    garbage.complete = lambda *a, **k: "no idea"
    assert judge_relation(garbage, "x", cands).relation == "NEW"
