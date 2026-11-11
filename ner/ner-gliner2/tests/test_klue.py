from config.klue import classify_errors, sample_train, score, to_example

TAG_NAMES = ['B-DT', 'I-DT', 'B-LC', 'I-LC', 'B-OG', 'I-OG', 'B-PS', 'I-PS',
             'B-QT', 'I-QT', 'B-TI', 'I-TI', 'O']


def test_to_example_char_bio():
    row = {"tokens": list("이재용 회장이"), "ner_tags": [6, 7, 7, 12, 12, 12, 12]}
    assert to_example(row, TAG_NAMES) == ("이재용 회장이", [(0, 3, "person")])


def test_score_exact_match_only():
    data = [("A사 B", [(0, 2, "organization"), (3, 4, "person")])]
    preds = [{(0, 2, "organization"), (3, 4, "location")}]
    s = score(preds, data)
    assert s["overall"]["precision"] == 0.5 and s["overall"]["recall"] == 0.5
    assert s["organization"]["f1"] == 1.0


def test_classify_errors():
    data = [("이재용 회장 서울", [(0, 3, "person"), (7, 9, "location")])]
    preds = [{(0, 6, "person"), (7, 9, "organization")}]
    assert classify_errors(preds, data) == {
        "wider": 1, "narrower": 0, "type": 1, "spurious": 0, "missed": 0}


def test_classify_errors_spurious_and_missed():
    data = [("가나 다라", [(0, 2, "person")])]
    preds = [{(3, 5, "location")}]
    assert classify_errors(preds, data) == {
        "wider": 0, "narrower": 0, "type": 0, "spurious": 1, "missed": 1}


def test_sample_train_is_seeded():
    rows = [{"i": i} for i in range(100)]
    assert sample_train(rows, 5, 42) == sample_train(rows, 5, 42)
