from config.security_audit.ontology import load_ontology


def test_counts():
    o = load_ontology()
    assert len(o.entity_types) == 14 and len(o.extract_labels()) == 11
    assert len(o.relations) == 17 and sum(r.extract for r in o.relations) == 10
    assert {"incident", "finding", "action"} <= set(o.records) and len(o.records) == 13


def test_allowed():
    o = load_ontology()
    assert o.allowed("HOLDS_POSITION", "Person", "Position")
    assert not o.allowed("HOLDS_POSITION", "Unit", "Position")
    assert not o.allowed("NOT_A_RELATION", "Person", "Unit")


def test_incident_type_choices():
    f = {f.name: f for f in load_ontology().records["incident"].fields}
    assert len(f["유형"].choices) == 6 and f["관련자"].dtype == "list"


def test_extract_labels_have_korean_descriptions():
    labels = load_ontology().extract_labels()
    assert "SecurityIncident" not in labels and labels["Position"]
