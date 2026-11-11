# LLM 증류 실험 (experiments/03) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** KLUE train 2,000문장에 gpt-5.4-mini가 단 정답으로 GLiNER2 LoRA 어댑터를 학습하고, 기본·KLUE 어댑터·증류 어댑터·gpt-5.4-mini를 KLUE 검증 5,000문장에서 비교한다. 자원·비용도 측정한다.

**Architecture:** 순수 로직은 `config/`에 둔다.
- `config/klue.py`: KLUE 데이터 로딩·변환·채점·오류 분류
- `config/teacher.py`: 선생님 프롬프트, 응답 스키마, 위치 찾기, 비동기 호출

실행은 `scripts/run_llm_distillation.py` 하나로 하고, 단계별 캐시를 `outputs/distill/`에 저장한다. 그래서 중간에 멈춰도 이어서 실행할 수 있다. 최종 수치는 `experiments/results/llm_distillation.json`에 저장하고, `experiments/03_llm_distillation.ipynb`는 이 JSON을 불러와 분석만 한다.

**Tech Stack:** Python 3.14, uv, gliner2 2.1.0, kiwipiepy, openai(AsyncOpenAI Responses API), pytest(`uv run --with pytest`로 일회성 실행, 의존성에 추가하지 않음)

**Spec:** `ner/ner-gliner2/docs/2026-11-11-llm-distillation-design.md`

## Global Constraints

- 선생님 모델: `.env`의 `LLM_MODEL` (= `gpt-5.4-mini`). 설정은 Responses API, 추론 강도 기본값, 실시간 호출이다. Batch API는 쓰지 않는다.
- 레이블은 `person`, `organization`, `location` 세 개만 쓴다. KLUE의 `PS→person`, `OG→organization`, `LC→location`이다.
- 학습 문장은 `random.seed(42)` 직후 `random.sample(train_rows, 2000)`으로 뽑는다. `notebooks/02`와 같은 2,000문장이어야 한다.
- 평가 데이터는 KLUE validation 5,000문장 전체다. 채점은 범위와 타입이 정확히 일치해야 정답으로 센다(`notebooks/02`의 `evaluate`와 같은 정의). 기본 threshold는 0.5다.
- 학습 설정은 `fastino/gliner2.5-multi-v1`, `kiwi_splitter`, `TrainingConfig(use_lora=True, save_adapter_only=True, num_epochs=2, batch_size=8, seed=42, eval_strategy="no", num_workers=0, pin_memory=False)`이다. 나머지는 기본값을 쓴다.
- 선생님 프롬프트에는 KLUE 규칙을 자세히 글로 설명한다. **few-shot 예시는 넣지 않는다.** KLUE 문장이나 실존 사례도 넣지 않는다. 직함을 떼는 식의 KLUE 맞춤 후처리도 하지 않는다.
- 비용은 실제 토큰 수 × 실행 시점 단가로 계산한다. 단가와 출처 URL을 결과에 함께 기록한다. 로컬 CPU 시간은 금액으로 바꾸지 않는다.
- 커밋하지 않는다(사용자가 요청할 때만 한다). 기존 `notebooks/`는 수정하지 않는다.
- 노트북 출력은 제거한 상태로 저장한다. 마크다운은 기존 노트북처럼 한국어 서술체("~한다")로 쓴다.

## Review Focus

1. **선생님이 원문에 없는 표면형을 반환하는 경우**(정규화, 띄어쓰기 변경, 조사 포함): 해당 엔티티를 버리고 `dropped` 개수에 센다. 학습은 계속 진행한다.
2. **같은 문자열이 한 문장에 여러 번 나오는 경우**: 선생님이 같은 이름을 n번 반환하면 앞에서부터 n개 위치를 쓴다. 한 번만 반환하면 첫 위치만 쓴다.
3. **API 실패, 타임아웃, 스키마 위반**: 재시도한 뒤에도 실패하면 그 문장을 `failed`에 기록한다. 학습에서는 그 문장을 빼고, 평가에서는 예측이 빈 것으로 처리한다. 실행 전체는 멈추지 않는다.
4. **실행 중단 후 재실행**: 캐시된 문장은 다시 호출하지 않는다. 비용도 두 번 세지 않는다.
5. **선생님 위치가 Kiwi 경계와 맞지 않는 경우**: 학습 변환에서 제외하고 그 비율을 기록한다(`notebooks/02`와 같은 규칙).

---

### Task 1: KLUE 데이터·채점 모듈

**Files:**
- Create: `config/klue.py`
- Test: `tests/test_klue.py`

**Interfaces:**
- Produces:
  - `LABELS: list[str]`, `KLUE_TO_LABEL: dict[str, str]`
  - `Span = tuple[int, int, str]`
  - `load_klue(split: str) -> tuple[list[dict], list[str]]`: `notebooks/02` cell 9와 같다.
  - `to_example(row: dict, tag_names: list[str]) -> tuple[str, list[Span]]`: 같은 동작이다.
  - `sample_train(train_rows, size=2000, seed=42) -> list[dict]`: `random.seed(seed)` 후 `random.sample`.
  - `to_input_example(text: str, spans: list[Span]) -> InputExample`: 모든 레이블을 빈 목록으로 선언하고, Kiwi 경계 밖 범위는 제외한다.
  - `aligned_ratio(data: list[tuple[str, list[Span]]]) -> tuple[int, int]`: (경계 일치 수, 전체 수)
  - `score(predictions: list[set[Span]], data: list[tuple[str, list[Span]]]) -> dict`: `notebooks/02`의 `evaluate`와 같은 형식의 `{label|overall: {precision, recall, f1}, "particles": int}`. 시간은 포함하지 않는다.
  - `classify_errors(predictions, data) -> dict[str, int]`: 키는 `wider`, `narrower`, `type`, `spurious`, `missed`.
  - `gliner_predict(model, texts: list[str], threshold=0.5) -> tuple[list[set[Span]], float]`: (예측, 총 초)

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/test_klue.py`)

```python
def test_to_example_char_bio():
    row = {"tokens": list("이재용 회장이"), "ner_tags": [6, 7, 7, 12, 12, 12, 12]}
    names = ['B-DT','I-DT','B-LC','I-LC','B-OG','I-OG','B-PS','I-PS','B-QT','I-QT','B-TI','I-TI','O']
    assert to_example(row, names) == ("이재용 회장이", [(0, 3, "person")])

def test_score_exact_match_only():
    data = [("A사 B", [(0, 2, "organization"), (3, 4, "person")])]
    preds = [{(0, 2, "organization"), (3, 4, "location")}]
    s = score(preds, data)
    assert s["overall"]["precision"] == 0.5 and s["overall"]["recall"] == 0.5
    assert s["organization"]["f1"] == 1.0

def test_classify_errors():
    data = [("이재용 회장 서울", [(0, 3, "person"), (7, 9, "location")])]
    preds = [{(0, 6, "person"), (7, 9, "organization")}]
    assert classify_errors(preds, data) == {"wider": 1, "narrower": 0, "type": 1, "spurious": 0, "missed": 0}

def test_sample_train_is_seeded():
    rows = [{"i": i} for i in range(100)]
    assert sample_train(rows, 5, 42) == sample_train(rows, 5, 42)
```

`classify_errors`는 오답을 한 번씩만 센다. 우선순위는 다음과 같다.
- 같은 범위에 타입만 다르면 → `type`
- 같은 타입의 정답 범위를 포함하면 → `wider`
- 같은 타입의 정답 범위 안에 들어가면 → `narrower`
- 그 외 예측 오답 → `spurious`
- 어떤 예측과도 겹치지 않는 정답 → `missed`

위에서 이미 짝지어진 정답은 `missed`로 세지 않는다.

- [ ] **Step 2: 실패 확인**: `uv run --with pytest pytest tests/test_klue.py -q` → `ImportError`
- [ ] **Step 3: `config/klue.py` 구현**: `notebooks/02` cell 9, 15, 17의 로직을 함수로 옮긴다. `has_particle`은 `config.extractors.kiwi`를 사용한다.
- [ ] **Step 4: 통과 확인**: 같은 명령 → 4 passed

### Task 2: 선생님 라벨러

**Files:**
- Create: `config/teacher.py`
- Test: `tests/test_teacher.py`

**Interfaces:**
- Consumes: `LABELS`, `Span` (Task 1)
- Produces:
  - `TEACHER_PROMPT: str`: KLUE 규칙을 자세히 서술한다. few-shot 예시는 넣지 않는다. 다룰 항목은 spec의 「선생님 라벨링」 목록 전부다.
  - `RESPONSE_SCHEMA: dict`: `{"entities": [{"text": str, "type": enum(LABELS)}]}`, strict.
  - `locate_spans(text: str, entities: list[dict]) -> tuple[list[Span], int]`: (위치를 붙인 범위, 버린 개수)
  - `class TeacherLabeler(model: str, concurrency: int = 16, max_retries: int = 3)`
    - `async label_many(texts: list[str], cache_path: Path) -> list[dict]`
    - 문장별 레코드: `{"text", "entities", "spans": list[list], "dropped", "input_tokens", "output_tokens", "seconds", "error": str|None}`
    - `cache_path`는 JSONL이다. 이미 있는 `text`는 다시 호출하지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_locate_repeated_names():
    spans, dropped = locate_spans("KT와 KT가", [{"text": "KT", "type": "organization"}] * 2)
    assert spans == [(0, 2, "organization"), (4, 6, "organization")] and dropped == 0

def test_locate_missing_surface_is_dropped():
    spans, dropped = locate_spans("서울시청", [{"text": "서울 시청", "type": "location"}])
    assert spans == [] and dropped == 1

def test_prompt_has_no_fewshot():
    assert "이재용" not in TEACHER_PROMPT and "예시 문장" not in TEACHER_PROMPT
    for rule in ("직함", "조사", "집단", "문맥"):
        assert rule in TEACHER_PROMPT

def test_cache_skips_done(tmp_path, monkeypatch):
    # label_many: 캐시에 있는 문장은 API를 호출하지 않는다 (_call_one을 monkeypatch로 카운트)
    ...
```

마지막 테스트는 실제 assert를 쓴다. 캐시 파일에 문장 A를 미리 써 두고 `label_many(["A", "B"])`를 실행하면, `_call_one` 호출이 1회여야 한다.

- [ ] **Step 2: 실패 확인**: `uv run --with pytest pytest tests/test_teacher.py -q` → `ImportError`
- [ ] **Step 3: `config/teacher.py` 구현**
  - `_call_one(text)`는 `responses.create(model, instructions=TEACHER_PROMPT, input=text, text={"format": json_schema strict})`를 호출한다.
  - 토큰은 `response.usage.input_tokens/output_tokens`에서 가져온다.
  - 동시성은 `asyncio.Semaphore(concurrency)`로 제한한다.
  - 실패하면 지수 백오프로 재시도한다.
  - 결과는 캐시에 한 줄씩 추가하고 즉시 flush한다.
- [ ] **Step 4: 통과 확인**: 4 passed
- [ ] **Step 5: 실 API 스모크 테스트**: KLUE train 첫 3문장을 라벨링해서 출력과 토큰을 눈으로 확인한다. 3문장 모두 `error`가 None이어야 한다. 출력이 프롬프트 규칙(조사·직함 제외)을 따르는지도 확인한다.

### Task 3: 실행 스크립트

**Files:**
- Create: `scripts/run_llm_distillation.py`

**Interfaces:**
- Consumes: Task 1, Task 2 전부
- Produces: `experiments/results/llm_distillation.json`. 노트북이 이 구조를 그대로 읽는다.

```text
{"run": {date, model, price: {input_per_mtok, output_per_mtok, source}, device, train_size, eval_size},
 "teacher_vs_human_train": score dict,            # 학습 2,000문장: 선생님 정답 vs 사람 정답
 "labeling": {sentences, failed, dropped, aligned, total_spans, input_tokens, output_tokens, cost_usd, wall_seconds, mean_call_seconds},
 "training": {seconds, samples_per_second, trainable_params},
 "conditions": {"base": score+seconds, "klue_lora": ..., "distill_lora": ...},
 "errors": {"klue_lora": classify dict, "distill_lora": ..., "teacher_train": classify dict (학습 문장, 사람 정답 대비)},
 "examples": [{"text", "gold", "base", "klue_lora", "distill_lora"}],   # 6~8개
 "breakeven_sentences": int}  # 손익분기: 라벨링 비용 ÷ 문장당 LLM 비용(라벨링 측정값) = 2,000 이므로 대신 '라벨링 비용으로 LLM 추론 가능한 문장 수'와 GLiNER 추론 시간 대비 표를 둔다
```

단계와 캐시는 아래와 같다. 각 단계는 산출물이 이미 있으면 건너뛴다.
1. `outputs/distill/train_labels.jsonl`: 선생님이 학습 2,000문장을 라벨링한 결과
2. `outputs/distill/lora/final`: 증류 어댑터. `ExtractorTrainer`로 학습하고 시간을 기록한다.
3. `outputs/distill/predictions.json`: 기본, KLUE 어댑터(`outputs/klue_ner_lora/final`), 증류 어댑터의 5,000문장 예측과 시간
4. 집계해서 결과 JSON을 저장한다.

`--limit N` 옵션은 학습·검증 문장 수를 N으로 줄여 스모크 테스트를 하고, 산출물은 `outputs/distill_smoke/`에 저장한다. 단가는 `--price-input`, `--price-output`(USD/1M tokens), `--price-source` 인자로 받는다. 단가 인자가 없으면 실행을 거부한다.

예시 문장은 검증 데이터에서 고른다. 증류 어댑터와 KLUE 어댑터의 결과가 다른 문장을 우선하고, 증류가 맞힌 경우와 틀린 경우를 섞는다. 선택 규칙은 구현자에게 맡긴다.

- [ ] **Step 1: 단가 확인**: 실행 시점의 gpt-5.4-mini 입력·출력 단가를 OpenAI 공식 가격 페이지에서 확인하고 URL을 기록한다.
- [ ] **Step 2: 스모크 실행**: `uv run python scripts/run_llm_distillation.py --limit 20 --price-input … --price-output … --price-source …` → `outputs/distill_smoke/` 아래에 결과 JSON이 생기고, 모든 키가 채워져야 한다.
- [ ] **Step 3: 재실행 캐시 확인**: 같은 명령을 다시 실행해서 API 호출 수 0, 비용 증가 0이 로그에 찍히는지 확인한다.

### Task 4: 전체 실행

- [ ] **Step 1**: `--limit` 없이 백그라운드로 실행한다. 예상 시간은 라벨링(2,000회 동시 호출) 약 5~10분, 학습 약 4분, 평가 3조건 × 약 4분이다.
- [ ] **Step 2: 검증**:
  - `conditions.klue_lora.overall.f1`이 0.812 ± 0.002여야 한다(기존 측정 재현).
  - `conditions.base.overall.f1`이 0.580 ± 0.01이어야 한다(CPU/GPU 차이 허용).
  - `labeling.failed`가 전체의 1% 미만이어야 한다.

### Task 5: 노트북과 README

**Files:**
- Create: `experiments/03_llm_distillation.ipynb`
- Modify: `README.md` (구성 표에 행 추가, 실행 방법에 스크립트 안내 추가)

**Interfaces:**
- Consumes: `experiments/results/llm_distillation.json` (Task 3 구조)

- [ ] **Step 1: 노트북 작성**: 셀 구성은 spec 「산출물」의 순서를 따른다.
  - 준비 셀은 기존 experiments 노트북 관례를 따른다. `PROJECT` 경로 찾기, `RESULT_PATH`를 쓴다.
  - `RUN_EXPERIMENT = False`면 JSON만 읽는다. `True`면 스크립트를 실행하라는 안내만 출력한다(노트북 안에서 7,000회 호출을 하지 않는다).
  - 결과 표: 네 조건 비교, 선생님 대 사람 일치도, 오류 분류, 자원·비용, 손익분기
  - 정성 비교: 예시 문장마다 정답/기본/KLUE/증류 결과를 나란히 보여준다.
  - 마크다운 관찰 기록은 실제 수치로 채운다.
  - 결론에는 다음 한계를 적는다. 단일 seed, 범위가 KLUE 도메인, 선생님 프롬프트는 KLUE 규칙을 알고 있음(few-shot은 없음), Batch API를 쓰면 비용이 약 절반이라는 점.
- [ ] **Step 2: 실행 확인**: 프로젝트 루트에서 노트북의 모든 코드 셀을 실행해서 오류가 없어야 한다. 마크다운 표의 열 개수를 검사한다.
- [ ] **Step 3: README 갱신**하고 사용자에게 보고한다(커밋하지 않음).
