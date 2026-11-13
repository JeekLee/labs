from config.security_audit.data import RecordGold
from config.security_audit.evaluate import choice_literal_split, score_by_group
from config.security_audit.ontology import load_ontology


def test_choice_literal_split():
    g = [RecordGold("incident", "비인가 출입 확인", {"유형": "비인가 출입"}),
         RecordGold("incident", "무단 진입 확인", {"유형": "비인가 출입"})]
    s = choice_literal_split([{"유형": "비인가 출입"}, {"유형": "보안교육 미이수"}], g, load_ontology())
    assert s["incident.유형"] == {"literal": 1.0, "absent": 0.0, "n_literal": 1, "n_absent": 1}


def test_choice_literal_split_handles_empty_bucket():
    g = [RecordGold("medium_transfer", "반출", {"구분": "반출"})]
    s = choice_literal_split([{"구분": "반출"}], g, load_ontology())
    assert s["medium_transfer.구분"] == {"literal": 1.0, "absent": None, "n_literal": 1, "n_absent": 0}


def test_score_by_group():
    items = [{"k": "a", "x": 1}, {"k": "b", "x": 0}, {"k": "a", "x": 1}]
    out = score_by_group(lambda xs: {"mean": sum(i["x"] for i in xs) / len(xs), "n": len(xs)}, items, "k")
    assert out == {"a": {"mean": 1.0, "n": 2}, "b": {"mean": 0.0, "n": 1}}
