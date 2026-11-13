from config.security_audit.forms import FORMS_DIR, load_forms
from config.security_audit.ontology import load_ontology

LEDGERS = {"document_entry", "loan", "reading", "clearance", "custodian", "entry", "daily_check",
           "medium_entry", "medium_transfer", "system_entry"}


def test_every_form_has_one_eval_only_variant_and_maps_record_fields():
    forms, o = load_forms(FORMS_DIR), load_ontology()
    assert set(forms) == LEDGERS
    for name, form in forms.items():
        assert sum(v.eval_only for v in form.variants) == 1 and 3 <= len(form.variants) <= 4
        assert all(v.id.endswith("_eval") == v.eval_only for v in form.variants)
        record_fields = {f.name for f in o.records[name].fields}
        mapped = {c.field for c in form.columns if c.field} | set(form.meta_fields)
        assert mapped >= record_fields - {"비고"}, (name, record_fields - mapped)
        assert mapped <= record_fields, (name, mapped - record_fields)


def test_eval_only_variant_differs_in_more_than_names():
    for form in load_forms(FORMS_DIR).values():
        ev = next(v for v in form.variants if v.eval_only)
        for other in (v for v in form.variants if not v.eval_only):
            assert ev.date_format != other.date_format or ev.grouped != other.grouped or ev.order != other.order


def test_variant_headers_cover_columns_and_order_is_permutation():
    for form in load_forms(FORMS_DIR).values():
        keys = [c.key for c in form.columns]
        for v in form.variants:
            assert sorted(v.order) == sorted(keys), (form.record, v.id)
            assert set(v.headers) == set(keys), (form.record, v.id)
            assert all(k in keys for group in v.grouped for k in group)
