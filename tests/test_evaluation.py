import json

from mini_supermemory import evaluation
from mini_supermemory.config import Settings
from mini_supermemory.evaluation import HeuristicLLM, score


def test_score_recall_and_stale():
    s = score(["User is switching to Puma", "User loves Adidas sneakers"], [["puma"]], ["adidas"])
    assert s["recall"] == 1.0 and s["stale"] is True
    # An item mentioning both the old and the new fact is not counted as stale.
    s = score(["User switched from Adidas to Puma"], [["puma"]], ["adidas"])
    assert s["stale"] is False
    # Keywords match at word starts ("swim" matches "swimming") but not inside words.
    assert score(["User loves swimming"], [["swim"]], [])["recall"] == 1.0
    assert score(["User has a dachshund"], [["dog"], ["hund"]], [])["recall"] == 0.0
    assert score(["anything"], [], ["x"])["recall"] is None


def test_dataset_is_well_formed():
    data = json.loads(evaluation.DATASET.read_text())
    assert 20 <= len(data["scenarios"]) <= 30
    assert {s["category"] for s in data["scenarios"]} == set(evaluation.CATEGORIES)
    assert len({s["id"] for s in data["scenarios"]}) == len(data["scenarios"])


def test_heuristic_llm_extracts_expiry():
    out = json.loads(HeuristicLLM().complete("You extract ...", [{"role": "user", "content": '"""\nI have an exam tomorrow\n"""'}]))
    assert out["memories"][0]["expires_in_hours"] == 48


def test_offline_run_writes_results(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "RESULTS_DIR", tmp_path)
    result = evaluation.run(Settings(embedding_backend="hash"), offline=True, limit=4, workers=2, label="t")
    assert result["n_scenarios"] == 4 and "overall" in result["summary"]
    assert (tmp_path / "t.json").exists() and (tmp_path / "t.md").read_text().startswith("**Provider")
    assert evaluation.load_results()[0]["label"] == "t"
