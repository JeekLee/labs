from config.security_audit.forms import FORMS_DIR, load_forms
from config.security_audit.ontology import load_ontology
from scripts.security_audit.make_scenarios import load_dict, load_dicts, make_scenarios

SPLITTABLE = ("names", "units", "facilities", "documents", "regulations", "medium_prefixes", "system_prefixes")


def scenarios():
    return make_scenarios(45, 15, 42, load_dicts(), load_ontology())


def test_dictionaries_disjoint():
    tr, ev = load_dict("train"), load_dict("eval")
    for kind in SPLITTABLE:
        assert not set(tr[kind]) & set(ev[kind]), kind
    assert tr["credential_prefix"] != ev["credential_prefix"]


def test_train_has_no_eval_surfaces_or_variants():
    ev = load_dict("eval")
    eval_surfaces = {s for k in ("names", "units", "facilities", "documents", "regulations") for s in ev[k]}
    eval_prefixes = tuple(ev["medium_prefixes"] + ev["system_prefixes"] + [ev["credential_prefix"]])
    for s in scenarios():
        if s["split"] != "train":
            continue
        assert not any(l["variant_id"].endswith("_eval") for l in s["ledgers"])
        for e in s["entities"]:
            assert not any(name in e["surface"] for name in eval_surfaces), e
            assert not e["surface"].startswith(eval_prefixes), e


def test_eval_mixes_seen_and_unseen():
    sc = [s for s in scenarios() if s["split"] == "eval"]
    ledgers = [l for s in sc for l in s["ledgers"]]
    share = sum(l["variant_id"].endswith("_eval") for l in ledgers) / len(ledgers)
    assert 0.3 <= share <= 0.7
    tr_names = set(load_dict("train")["names"])
    people = [e for s in sc for e in s["entities"] if e["type"] == "Person"]
    seen = sum(any(n in p["surface"] for n in tr_names) for p in people) / len(people)
    assert 0.3 <= seen <= 0.7


def test_counts_rows_and_ledger_assignment():
    sc = scenarios()
    assert len(sc) == 60 and sum(s["split"] == "eval" for s in sc) == 15
    forms = load_forms(FORMS_DIR)
    for s in sc:
        assert 2 <= len(s["ledgers"]) <= 4
        for l in s["ledgers"]:
            assert l["record"] in forms and 6 <= len(l["rows"]) <= 10


def test_seeded():
    assert scenarios() == scenarios()


def test_surface_types_consistent():
    seen = {}
    for s in scenarios():
        for e in s["entities"]:
            assert seen.setdefault(e["surface"], e["type"]) == e["type"], e


def test_dates_and_times_in_column_order():
    forms = load_forms(FORMS_DIR)
    for s in scenarios():
        for ledger in s["ledgers"]:
            cols = forms[ledger["record"]].columns
            for kinds in ({"date", "datetime"}, {"time"}):
                keys = [c.key for c in cols if c.kind in kinds]
                for row in ledger["rows"]:
                    values = [row[k]["surface"] for k in keys if row[k]["surface"]]
                    assert values == sorted(values), (ledger["record"], values)
