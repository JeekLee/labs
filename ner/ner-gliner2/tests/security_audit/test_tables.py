import json

from config.security_audit.annotation import parse_marked, validate_document
from config.security_audit.forms import FORMS_DIR, load_forms
from config.security_audit.ontology import load_ontology
from config.security_audit.tables import format_date, render_ledger

P = lambda s: {"surface": s, "type": "Person"}
U = lambda s: {"surface": s, "type": "Unit"}
F = lambda s: {"surface": s, "type": "Facility"}
M = lambda s: {"surface": s, "type": "StorageMedium"}
V = lambda s: {"surface": s, "type": None}
IDS = dict(doc_id="d1", scenario_id="s1", split="train", area="정보통신보안")


def sample_transfer(blank_confirm=False):
    meta = {"부대": U("제9013보병사단 정보통신대대"), "관리책임자": P("윤지환 하사")}
    rows = [
        {"반출입 일시": V("2026-04-07 09:30"), "구분": V("반출"), "장비명": V("업무용 USB"), "관리번호": M("USB-118"),
         "사용자": P("장지후 병장"), "용도": V("정비 자료 이동"), "점검사항": V("자료 삭제 확인"), "확인자": P("윤 하사"), "비고": V("")},
        {"반출입 일시": V("2026-04-08 17:10"), "구분": V("반입"), "장비명": V("업무용 USB"), "관리번호": M("USB-118"),
         "사용자": P("장지후 병장"), "용도": V("정비 자료 이동"), "점검사항": V("악성코드 검사 완료"),
         "확인자": V("") if blank_confirm else P("윤 하사"), "비고": V("")},
    ]
    return meta, rows


def test_same_facts_same_gold_across_variants():
    form = load_forms(FORMS_DIR)["medium_transfer"]
    meta, rows = sample_transfer()
    docs = [render_ledger(form, v, meta, rows, **IDS) for v in form.variants]
    dates = {"반출입 일시"}  # rendered in each variant's own date format
    norm = lambda d: (sorted(json.dumps({k: r[k] for k in ("head", "type", "tail")}, ensure_ascii=False)
                             for r in d["relations"]),
                      [{k: v for k, v in r["fields"].items() if k not in dates} for r in d["records"]],
                      [sorted(r["fields"]) for r in d["records"]])
    keyed = [norm(d) for d in docs]
    assert all(k == keyed[0] for k in keyed)
    assert all(validate_document(d, load_ontology()) == [] for d in docs)
    assert {"head": "윤지환 하사", "type": "RESPONSIBLE_FOR", "tail": "USB-118"} in [
        {k: r[k] for k in ("head", "type", "tail")} for r in docs[0]["relations"]]


def test_grouped_cell_spans():
    form = load_forms(FORMS_DIR)["entry"]
    v = next(v for v in form.variants if ["소속", "출입자"] in v.grouped)
    meta = {"부대": U("제9013보병사단 정보통신대대"), "구역": F("제1통제구역"), "관리책임자": P("윤지환 하사")}
    rows = [{"일자": V("2026-04-07"), "출입시간": V("09:00"), "퇴청시간": V("10:20"), "용무": V("장비 점검"),
             "소속": U("제9013보병사단 정보통신대대"), "출입자": P("병장 장지후"), "입회자": P("윤 하사"), "비고": V("")}]
    doc = render_ledger(form, v, meta, rows, **IDS)
    text, spans = parse_marked(doc["paragraphs"][0])
    found = {(text[s:e], t) for s, e, t in spans}
    assert ("병장 장지후", "Person") in found and ("제9013보병사단 정보통신대대", "Unit") in found
    assert {"head": "병장 장지후", "type": "BELONGS_TO", "tail": "제9013보병사단 정보통신대대"} in [
        {k: r[k] for k in ("head", "type", "tail")} for r in doc["relations"]]
    assert validate_document(doc, load_ontology()) == []


def test_blank_field_omitted_and_valid():
    form = load_forms(FORMS_DIR)["medium_transfer"]
    meta, rows = sample_transfer(blank_confirm=True)
    doc = render_ledger(form, form.variants[0], meta, rows, **IDS)
    assert "확인자" not in doc["records"][1]["fields"]
    assert doc["records"][1]["fields"]["관리책임자"] == ["윤지환 하사"]
    assert validate_document(doc, load_ontology()) == []


def test_format_date():
    assert format_date("2026-04-07", "dot") == "2026. 4. 7."
    assert format_date("2026-04-07", "dash") == "2026-04-07"
    assert format_date("2026-04-07", "short") == "26-04-07"
    assert format_date("2026-04-07", "korean") == "2026년 4월 7일"
    assert format_date("2026-04-07 09:30", "short") == "26-04-07 09:30"
