from config.security_audit.ontology import load_ontology
from config.security_audit.data import ParagraphGold, RecordGold
from config.security_audit.evaluate import score_entities, score_records, score_relations


def test_relation_triple_f1():
    g = [ParagraphGold("t", [], [("김 상병", "BELONGS_TO", "정보통신대대")])]
    s = score_relations([{("김 상병", "BELONGS_TO", "정보통신대대"), ("김 상병", "HOLDS", "출입증")}], g)
    assert s["overall"]["precision"] == 0.5 and s["overall"]["recall"] == 1.0
    assert s["BELONGS_TO"]["f1"] == 1.0


def test_record_fields_list_and_choice():
    g = [RecordGold("incident", "t", {"유형": "비인가 출입", "관련자": ["김 상병", "이 병장"]})]
    s = score_records([{"유형": "비인가 출입", "관련자": ["김 상병"]}], g, load_ontology())
    assert s["choice_accuracy"]["incident.유형"] == 1.0
    assert s["fields"]["incident.관련자"]["precision"] == 1.0 and s["fields"]["incident.관련자"]["recall"] == 0.5


def test_record_string_field_and_missing_prediction():
    g = [RecordGold("incident", "t", {"일자": "2026. 3. 12."}), RecordGold("incident", "t", {"유형": "비인가 출입"})]
    s = score_records([{"일자": "2026. 3. 12."}, {}], g, load_ontology())
    assert s["fields"]["incident.일자"]["f1"] == 1.0 and s["choice_accuracy"]["incident.유형"] == 0.0


def test_entities_use_given_labels():
    g = [ParagraphGold("김 상병", [(0, 4, "Person")], [])]
    s = score_entities([{(0, 4, "Person")}], g, ["Person", "Unit"])
    assert s["Person"]["f1"] == 1.0 and s["overall"]["f1"] == 1.0


def test_klue_score_unchanged_default():
    from config.klue import score
    assert score([{(0, 2, "person")}], [("가나", [(0, 2, "person")])])["overall"]["f1"] == 1.0
