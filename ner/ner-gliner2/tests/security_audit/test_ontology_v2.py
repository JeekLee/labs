from config.security_audit.ontology import load_ontology

LEDGER_RECORDS = {"document_entry", "loan", "reading", "clearance", "custodian", "entry", "daily_check",
                  "medium_entry", "medium_transfer", "system_entry"}


def test_v2_records_and_scopes():
    o = load_ontology()
    assert LEDGER_RECORDS | {"incident", "finding", "action"} == set(o.records)
    ledger = {r.type for r in o.relations if r.extract and "ledger" in r.scopes}
    assert ledger == {"BELONGS_TO", "HOLDS_POSITION", "POSITION_OF", "HOLDS", "RESPONSIBLE_FOR", "LOCATED_AT"}
    narrative_only = {r.type for r in o.relations if r.extract and r.scopes == ("narrative",)}
    assert narrative_only == {"ACCESSED", "CARRIED_OUT", "STORED_ON", "INSPECTED"}


def test_shared_role_names_have_one_description():
    o = load_ontology()
    descs = {}
    for r in o.records.values():
        for f in r.fields:
            if f.name in {"승인자", "확인자", "입회자", "보관책임자", "처리자", "관리책임자"}:
                descs.setdefault(f.name, set()).add(f.description_ko)
    assert descs and all(len(v) == 1 for v in descs.values())


def test_joint_schema_scope_filters_relations():
    class Engine:
        def create_schema(self):
            return Schema()

    class Schema:
        def __init__(self):
            self.relations = []

        def entity(self, *a):
            return self

        def relation(self, name, *a, **k):
            self.relations.append(name)
            return self

    o = load_ontology()
    assert set(o.joint_schema(Engine(), scope="ledger").relations) == {
        "BELONGS_TO", "HOLDS_POSITION", "POSITION_OF", "HOLDS", "RESPONSIBLE_FOR", "LOCATED_AT"}
    assert "CARRIED_OUT" in o.joint_schema(Engine(), scope="narrative").relations
