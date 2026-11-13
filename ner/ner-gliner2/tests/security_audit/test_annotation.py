from config.security_audit.annotation import parse_marked, validate_corpus, validate_document
from config.security_audit.ontology import load_ontology


def make_doc(paragraphs, relations=(), records=(), doc_id="d1", scenario_id="s1", split="train"):
    return {"doc_id": doc_id, "scenario_id": scenario_id, "split": split, "doc_type": "보안사고 경위서",
            "area_tags": ["문서보안"], "paragraphs": list(paragraphs), "relations": list(relations),
            "records": list(records)}


def test_parse_marked():
    text, spans = parse_marked("[[Person|김OO 상병]]이 [[StorageMedium|USB-17]]을 반출")
    assert text == "김OO 상병이 USB-17을 반출"
    assert spans == [(0, 6, "Person"), (8, 14, "StorageMedium")]


def test_relation_must_be_typed_and_present():
    doc = make_doc(["[[Person|김 상병]]이 [[Unit|정보통신대대]] 소속"],
                   relations=[{"paragraph": 0, "head": "김 상병", "type": "HOLDS", "tail": "정보통신대대"}])
    assert any("HOLDS" in e for e in validate_document(doc, load_ontology()))


def test_valid_document_has_no_errors():
    doc = make_doc(["[[Person|김 상병]]이 [[Unit|정보통신대대]] 소속으로 [[StorageMedium|USB-17]]을 반출"],
                   relations=[{"paragraph": 0, "head": "김 상병", "type": "BELONGS_TO", "tail": "정보통신대대"}],
                   records=[{"record": "incident", "text_paragraphs": [0],
                             "fields": {"유형": "저장매체 무단 반출", "관련자": ["김 상병"], "매체": ["USB-17"]}}])
    assert validate_document(doc, load_ontology()) == []


def test_record_field_value_must_appear_in_text():
    doc = make_doc(["[[Person|김 상병]]이 반출"],
                   records=[{"record": "incident", "text_paragraphs": [0],
                             "fields": {"유형": "저장매체 무단 반출", "관련자": ["이 병장"]}}])
    assert any("이 병장" in e for e in validate_document(doc, load_ontology()))


def test_corpus_type_consistency():
    a = make_doc(["[[Facility|문서보관실]] 점검"], doc_id="a")
    b = make_doc(["[[Unit|문서보관실]] 점검"], doc_id="b")
    assert any("문서보관실" in e for e in validate_corpus([a, b], load_ontology()))


def test_unknown_type_and_choice_rejected():
    doc = make_doc(["[[Rank|상병]]"], records=[{"record": "incident", "text_paragraphs": [0],
                                                 "fields": {"유형": "기타"}}])
    errors = validate_document(doc, load_ontology())
    assert any("Rank" in e for e in errors) and any("기타" in e for e in errors)


def test_malformed_markup_rejected():
    doc = make_doc(["[[Person|김 상병 이 반출"])
    assert any("markup" in e for e in validate_document(doc, load_ontology()))
