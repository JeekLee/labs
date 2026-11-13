import json

from config.security_audit.annotation import parse_marked
from config.security_audit.forms import FORMS_DIR, load_forms
from config.security_audit.storage import doc_to_files, files_to_doc
from config.security_audit.tables import render_ledger
from tests.security_audit.test_tables import IDS, sample_transfer


def narrative_doc():
    return {"doc_id": "s1-N1", "scenario_id": "s1", "split": "train", "doc_type": "보안사고 경위서",
            "doc_kind": "narrative", "area_tags": ["정보통신보안"],
            "paragraphs": ["1. [[Person|장지후 병장]]이 [[StorageMedium|USB-118]]을 반출함.",
                           "2. 관리 책임: [[Person|윤지환 하사]]가 [[StorageMedium|USB-118]]을 관리함."],
            "relations": [{"paragraph": 0, "head": "장지후 병장", "type": "CARRIED_OUT", "tail": "USB-118"},
                          {"paragraph": 1, "head": "윤지환 하사", "type": "RESPONSIBLE_FOR", "tail": "USB-118"}],
            "records": [{"record": "incident", "text_paragraphs": [0, 1],
                         "fields": {"유형": "저장매체 무단 반출", "관련자": ["장지후 병장"]}}]}


def plain(paragraph):
    return parse_marked(paragraph)


def test_narrative_roundtrip_and_markdown_has_no_markup():
    doc = narrative_doc()
    md, labels = doc_to_files(doc)
    assert "[[" not in md and md.startswith("# 보안사고 경위서")
    back = files_to_doc(md, labels)
    assert [plain(p) for p in back["paragraphs"]] == [plain(p) for p in doc["paragraphs"]]
    assert back["relations"] == doc["relations"] and back["records"] == doc["records"]
    for e in labels["entities"]:
        assert md[e["start"]:e["end"]] == e["text"]


def test_ledger_markdown_has_single_header_and_units_rebuild_context():
    form = load_forms(FORMS_DIR)["medium_transfer"]
    doc = render_ledger(form, form.variants[0], *sample_transfer(), **IDS)
    md, labels = doc_to_files(doc)
    lines = md.split("\n")
    assert sum(l.startswith("|---") for l in lines) == 1 and sum(l.startswith("| 2026.") for l in lines) == 2
    back = files_to_doc(md, labels)
    assert [plain(p) for p in back["paragraphs"]] == [plain(p) for p in doc["paragraphs"]]
    assert back["relations"] == doc["relations"] and back["records"] == doc["records"]
    assert back["variant_id"] == doc["variant_id"] and back["doc_kind"] == "ledger"
    json.dumps(labels, ensure_ascii=False)
