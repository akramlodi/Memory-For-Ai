from elephantus.extraction import ExtractedFact, extract_facts, parse_json_object
from tests.fakes import FakeLLM


def test_parse_json_tolerates_fences_and_chatter():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here you go: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}


def test_multi_fact_message_creates_separate_memories(engine, fake_llm):
    msg = "I study CS at MIT and this week I'm building a chess engine in Rust."
    fake_llm.facts[msg] = [
        {"text": "User studies computer science at MIT", "kind": "static", "expires_in_hours": None},
        {"text": "User is building a chess engine in Rust", "kind": "dynamic", "expires_in_hours": 168},
    ]
    result = engine.add(msg, "khan")
    assert [m["decision"] for m in result["memories"]] == ["NEW", "NEW"]
    mems = {m["text"]: m for m in engine.list_memories("khan")}
    assert mems["User studies computer science at MIT"]["kind"] == "static"
    rust = mems["User is building a chess engine in Rust"]
    assert rust["kind"] == "dynamic" and rust["expires_at"] is not None
    assert rust["source_document_id"] == result["id"]


def test_malformed_output_is_retried(engine, fake_llm):
    fake_llm.raw_extraction = ["oops not json", '{"memories": [{"text": "User likes tea", "kind": "static"}]}']
    result = engine.add("I like tea", "khan")
    assert [m["fact"] for m in result["memories"]] == ["User likes tea"]


def test_malformed_output_never_crashes(engine, fake_llm):
    fake_llm.raw_extraction = ["garbage", "still garbage"]
    result = engine.add("I like tea", "khan")
    assert result["memories"] == [] and engine.list_documents("khan")  # document still stored


def test_invalid_items_are_skipped():
    llm = FakeLLM(raw_extraction=[
        '{"memories": [{"text": "User likes tea", "kind": "STATIC"}, {"kind": "static"}, "User has a cat",'
        ' {"text": "User has an exam", "kind": "weird"}, {"text": "User flies today", "expires_in_hours": "-3"}]}'
    ])
    facts = extract_facts(llm, "msg", 0)
    assert [f.text for f in facts] == ["User likes tea", "User has a cat", "User flies today"]
    assert facts[0].kind == "static" and facts[2].expires_in_hours is None


def test_llm_failure_keeps_document(engine, fake_llm):
    from elephantus.llm import LLMError

    def boom(*a, **k):
        raise LLMError("provider down")

    fake_llm.complete = boom
    result = engine.add("I like tea", "khan")
    assert "provider down" in result["error"] and len(engine.list_documents("khan")) == 1


def test_extracted_fact_model():
    assert ExtractedFact(text="User likes tea", expires_in_hours=0).expires_in_hours is None
