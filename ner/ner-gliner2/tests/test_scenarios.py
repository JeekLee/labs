from config.security_audit import load_ontology
from scripts.security_audit.make_scenarios import load_dictionary, make_scenarios


def scenarios():
    return make_scenarios(30, 42, load_dictionary(), load_ontology())


def test_balanced_and_legal():
    sc, o = scenarios(), load_ontology()
    assert len(sc) == 30 and sum(s["split"] == "eval" for s in sc) == 6
    assert len({s["area"] for s in sc}) == 4 and len({s["incident_type"] for s in sc}) == 6
    for s in sc:
        types = {e["id"]: e["type"] for e in s["entities"]}
        assert all(o.allowed(r["type"], types[r["head"]], types[r["tail"]]) for r in s["relations"])


def test_seeded():
    d, o = load_dictionary(), load_ontology()
    assert make_scenarios(30, 42, d, o) == make_scenarios(30, 42, d, o)


def test_meeting_minutes_only_in_eval():
    sc = scenarios()
    assert all("보안대책 회의록" not in s["doc_types"] for s in sc if s["split"] == "train")
    assert any("보안대책 회의록" in s["doc_types"] for s in sc if s["split"] == "eval")


def test_surfaces_unique_across_types():
    seen = {}
    for s in scenarios():
        for e in s["entities"]:
            assert seen.setdefault(e["surface"], e["type"]) == e["type"]


def test_incident_type_is_a_choice():
    choices = {f.name: f for f in load_ontology().records["incident"].fields}["유형"].choices
    assert all(s["incident_type"] in choices for s in scenarios())
