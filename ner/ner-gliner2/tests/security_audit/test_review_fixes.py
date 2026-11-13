"""Tests for the final-review fix pass (C1, I1-I4, M2, M4)."""

from config.security_audit.annotation import record_text
from config.security_audit.ontology import load_ontology
from config.security_audit.data import ParagraphGold, RecordGold, record_examples, to_training_examples
from config.security_audit.evaluate import (choice_literal_rate, incident_description_scores, score_entities_by_seen,
                                        score_records)
from tests.security_audit.test_annotation import make_doc


def test_training_examples_carry_descriptions():  # I1
    o = load_ontology()
    examples, _ = to_training_examples(
        [ParagraphGold("김 상병", [(0, 4, "Person")], [])],
        [RecordGold("incident", "김민수 상병이 반출함", {"유형": "저장매체 무단 반출", "관련자": ["김민수 상병"]})], o)
    assert examples[0].entity_descriptions == o.extract_labels()
    assert examples[1].structures[0].descriptions["관련자"]


def test_record_text_shared_separator():  # M2
    doc = make_doc(["가 [[Person|김 상병]]", "나"], records=[{"record": "finding", "text_paragraphs": [0, 1], "fields": {}}])
    assert record_text(doc, doc["records"][0]) == "가 김 상병\n나"
    assert record_examples(doc, load_ontology())[0].text == record_text(doc, doc["records"][0])


def test_record_scores_keyed_by_record_and_field():  # M4
    g = [RecordGold("incident", "t", {"위반 규정": ["A"]}), RecordGold("finding", "t", {"위반 규정": ["B"]})]
    s = score_records([{"위반 규정": ["A"]}, {}], g, load_ontology())
    assert s["fields"]["incident.위반 규정"]["f1"] == 1.0 and s["fields"]["finding.위반 규정"]["recall"] == 0.0


def test_entities_split_by_seen_surface():  # C1
    golds = [ParagraphGold("가나 다라", [(0, 2, "Person"), (3, 5, "Unit")], [])]
    s = score_entities_by_seen([{(0, 2, "Person")}], golds, {("가나", "Person")}, ["Person", "Unit"])
    assert s["seen"]["overall"]["recall"] == 1.0 and s["unseen"]["overall"]["recall"] == 0.0
    assert s["seen_gold"] == 1 and s["unseen_gold"] == 1


def test_choice_literal_rate():  # I2
    golds = [RecordGold("action", "재교육 실시", {"조치 유형": "재교육"}), RecordGold("action", "교육", {"조치 유형": "재교육"})]
    assert choice_literal_rate(golds, load_ontology())["action.조치 유형"] == 0.5


def test_trained_relations_counted_by_type():  # I3
    _, stats = to_training_examples([ParagraphGold("김 상병 작전과", [], [("김 상병", "BELONGS_TO", "작전과")])], [],
                                    load_ontology(), unalignable=lambda text, values: set())
    assert stats["relations_trained_by_type"] == {"BELONGS_TO": 1}


class FakeModel:
    def __init__(self, spans_by_text):
        self.spans_by_text = spans_by_text

    def extract_entities(self, text, labels, include_spans=True, **_):
        return {"entities": {"incident_description": [{"start": s, "end": e} for s, e in self.spans_by_text.get(text, [])]}}


def test_incident_description_predicts_all_paragraphs():  # I4
    doc = make_doc(["가나다라마", "바사아자차"])
    doc["doc_type"] = "보안사고 경위서"
    doc["incident_description"] = [[0, 0, 2]]
    model = FakeModel({"가나다라마": [(0, 2)], "바사아자차": [(1, 3)]})
    s = incident_description_scores(model, [doc])
    assert s["exact_precision"] == 0.5 and s["exact_recall"] == 1.0
