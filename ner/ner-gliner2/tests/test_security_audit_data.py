from config.security_audit import load_ontology
from config.security_audit_data import (ParagraphGold, RecordGold, paragraph_examples, record_examples,
                                        split_docs, to_training_examples)
from tests.test_annotation import make_doc


def test_paragraph_gold_from_marked():
    doc = make_doc(["[[Person|이OO 중사]]는 [[Position|보안담당관]]이다"],
                   relations=[{"paragraph": 0, "head": "이OO 중사", "type": "HOLDS_POSITION", "tail": "보안담당관"}])
    [p] = paragraph_examples(doc, load_ontology())
    assert p.spans == [(0, 6, "Person"), (8, 13, "Position")]
    assert p.relations == [("이OO 중사", "HOLDS_POSITION", "보안담당관")]


def test_record_gold_joins_paragraphs():
    doc = make_doc(["1. [[Person|김 상병]] 반출", "2. 기타"],
                   records=[{"record": "finding", "text_paragraphs": [0], "fields": {"영역": "문서보안", "대상": ["김 상병"]}}])
    [r] = record_examples(doc, load_ontology())
    assert r == RecordGold("finding", "1. 김 상병 반출", {"영역": "문서보안", "대상": ["김 상병"]})


def test_all_labels_declared_and_misaligned_dropped():
    examples, stats = to_training_examples([ParagraphGold("가나다라", [(1, 3, "Person")], [])], [], load_ontology())
    assert set(examples[0].entities) == set(load_ontology().extract_labels())
    assert stats["misaligned"] == 1 and examples[0].entities["Person"] == []


def test_record_example_has_structure_with_choice():
    _, stats = to_training_examples([], [RecordGold("incident", "김민수 상병이 보안USB-100을 반출하였다.",
                                                      {"유형": "저장매체 무단 반출", "관련자": ["김민수 상병"]})],
                                    load_ontology())
    assert stats["record_examples"] == 1


def test_split_by_scenario():
    docs = [make_doc(["x"], doc_id=f"d{i}", scenario_id=f"s{i % 10}") for i in range(40)]
    docs.append(make_doc(["y"], doc_id="e", scenario_id="e1", split="eval"))
    s = split_docs(docs)
    train_sc = {d["scenario_id"] for d in s["train"]}
    dev_sc = {d["scenario_id"] for d in s["dev"]}
    assert train_sc and dev_sc and not (train_sc & dev_sc)
    assert {d["scenario_id"] for d in s["eval"]} == {"e1"}
