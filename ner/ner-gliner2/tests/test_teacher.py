import asyncio
import json

from config.teacher import TEACHER_PROMPT, TeacherLabeler, locate_spans


def test_locate_repeated_names():
    spans, dropped = locate_spans("KT와 KT가", [{"text": "KT", "type": "organization"}] * 2)
    assert spans == [(0, 2, "organization"), (4, 6, "organization")] and dropped == 0


def test_locate_single_mention_uses_first_occurrence():
    spans, dropped = locate_spans("KT와 KT가", [{"text": "KT", "type": "organization"}])
    assert spans == [(0, 2, "organization")] and dropped == 0


def test_locate_missing_surface_is_dropped():
    spans, dropped = locate_spans("서울시청", [{"text": "서울 시청", "type": "location"}])
    assert spans == [] and dropped == 1


def test_prompt_has_no_fewshot():
    assert "이재용" not in TEACHER_PROMPT and "예시 문장" not in TEACHER_PROMPT
    for rule in ("직함", "조사", "집단", "문맥"):
        assert rule in TEACHER_PROMPT


def test_cache_skips_done(tmp_path, monkeypatch):
    cache = tmp_path / "labels.jsonl"
    cache.write_text(json.dumps({"text": "A", "entities": [], "spans": [], "dropped": 0,
                                 "input_tokens": 1, "output_tokens": 1, "seconds": 0.1,
                                 "error": None}) + "\n")
    labeler = TeacherLabeler.__new__(TeacherLabeler)
    labeler.concurrency, labeler.max_retries = 2, 1
    calls = []

    async def fake_request(text):
        calls.append(text)
        return '{"entities": []}', 2, 3

    monkeypatch.setattr(labeler, "_request", fake_request)
    records = asyncio.run(labeler.label_many(["A", "B"], cache))
    assert calls == ["B"]
    assert [r["text"] for r in records] == ["A", "B"]
    assert len(cache.read_text().splitlines()) == 2


def test_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    labeler = TeacherLabeler.__new__(TeacherLabeler)
    labeler.concurrency, labeler.max_retries = 1, 2

    async def boom(text):
        raise RuntimeError("schema violation")

    monkeypatch.setattr(labeler, "_request", boom)
    monkeypatch.setattr("config.teacher.BACKOFF_SECONDS", 0)
    [record] = asyncio.run(labeler.label_many(["X"], tmp_path / "c.jsonl"))
    assert record["error"] and record["spans"] == []


def test_locate_short_surface_not_inside_longer_entity():
    text = "울산해양경찰서는 배를 울산으로 옮겼다"
    spans, dropped = locate_spans(text, [{"text": "울산해양경찰서", "type": "organization"},
                                         {"text": "울산", "type": "location"}])
    assert sorted(spans) == [(0, 7, "organization"), (12, 14, "location")] and dropped == 0


def test_locate_prefers_word_start_over_inside_word():
    text = "양상이 바뀌자 이 감독이 웃었다"
    spans, _ = locate_spans(text, [{"text": "이", "type": "person"}])
    assert spans == [(8, 9, "person")]


def test_locate_order_independent_of_entity_order():
    text = "북한 국방위원회는 북한의 입장을 밝혔다"
    a, _ = locate_spans(text, [{"text": "북한", "type": "location"},
                               {"text": "북한 국방위원회", "type": "organization"}])
    assert sorted(a) == [(0, 8, "organization"), (10, 12, "location")]


def test_attempts_and_usage_summed_across_failures(tmp_path, monkeypatch):
    labeler = TeacherLabeler.__new__(TeacherLabeler)
    labeler.concurrency, labeler.max_retries = 1, 3
    monkeypatch.setattr("config.teacher.BACKOFF_SECONDS", 0)
    responses = iter([("{bad json", 10, 2), ('{"entities": []}', 10, 3)])

    async def fake_request(text):
        return next(responses)

    monkeypatch.setattr(labeler, "_request", fake_request)
    [record] = asyncio.run(labeler.label_many(["X"], tmp_path / "c.jsonl"))
    assert record["error"] is None and record["attempts"] == 2
    assert record["input_tokens"] == 20 and record["output_tokens"] == 5
    assert record["finished_at"] >= record["started_at"] and record["run_id"]
