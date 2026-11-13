# 군부대 보안 감사 SFT 1단계 — 시험 생성(30 시나리오) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 보안 감사 온톨로지를 정의하고, 30개 가상 시나리오로 정답이 달린 문서를 만들어 GLiNER2 통합 어댑터를 학습·평가하는 전체 파이프라인을 시험 규모로 끝까지 돌린다.

**Architecture:**
- 온톨로지 YAML을 단일 기준으로 삼는다. 로더가 YAML에서 GLiNER 레이블, JointIE 관계 스키마, 구조화 레코드 스키마를 만든다.
- 시나리오 그래프는 코드로 생성한다(시드 고정). 문서는 서브에이전트가 시나리오를 근거로 인라인 표시 형식으로 작성한다.
- 검증기를 통과한 문서만 학습·평가 데이터로 변환한다. 실행 스크립트 하나가 학습, 예측, 채점을 하고, 노트북은 결과 JSON만 분석한다.

**Tech Stack:** Python 3.14, uv, gliner2 2.1.0, kiwipiepy, PyYAML(필요 시 `uv add pyyaml`), pytest(`uv run --with pytest python -m pytest`)

**Spec:** `ner/ner-gliner2/docs/2026-11-13-security-audit-sft-design.md`

**범위:** 이 계획은 30개 시나리오 시험 생성까지이다. 전체 규모(학습 약 150, 평가 약 40 시나리오) 확장은 시험 결과를 사용자에게 보고한 뒤 별도로 진행한다.

## Global Constraints

- 모든 데이터는 가상이다. 실존 부대명·인명·비밀 내용을 쓰지 않으며, 보안 규정은 공개된 일반 수준의 표현만 쓴다.
- 외부 API를 쓰지 않는다. 학습·평가는 로컬 macOS CPU에서 한다.
- 체크포인트는 `fastino/gliner2.5-multi-v1`, 분할기는 `config.extractors.kiwi_splitter`를 쓴다. LoRA 설정은 `notebooks/02`와 같다(2 epoch, 배치 8, seed 42, `use_lora=True, save_adapter_only=True`).
- 엔티티 추출 대상은 11개 타입이다: Person, Position, Unit, Organization, Facility, Credential, ClassifiedDocument, StorageMedium, InformationSystem, Regulation, Inspection. SecurityIncident, Finding, CorrectiveAction은 문서 단위로 만들고 추출하지 않는다.
- 관계는 17개이며, 엔티티 사이 10개(BELONGS_TO, HOLDS_POSITION, POSITION_OF, HOLDS, RESPONSIBLE_FOR, ACCESSED, CARRIED_OUT, STORED_ON, LOCATED_AT, INSPECTED)만 관계 추출 대상이다. 나머지 7개는 레코드 필드로 표현한다.
- 표기 규칙:
  - Person은 계급 포함(`김OO 상병`, `상병 김OO`, `김 상병`)이고 소속은 제외한다.
  - Position은 직책 자체이다. Unit은 공식 부대명 전체를 하나로 표시한다.
  - Inspection은 특정되는 행사명만 표시한다.
  - 같은 표면형은 모든 문서에서 같은 타입이어야 한다.
- 레코드 단위는 문서 서식으로 주어진다. 경위서는 문서 전체, 점검표·결과보고서는 번호 붙은 발견 항목, 조치표는 행 하나가 레코드 하나이다.
- 채점: 엔티티는 범위+타입 정확 일치 F1, 관계는 (head 표면형, 관계, tail 표면형) 정확 일치 F1, 레코드 필드는 값 정확 일치(목록 필드는 원소 단위 F1), 선택형 필드는 정확도.
- 커밋하지 않는다. 기존 노트북을 생성 스크립트로 다시 만들지 않는다.

## Review Focus

1. **같은 표면형이 문서마다 다른 타입으로 표시된 경우**: 검증기가 코퍼스 전체에서 찾아 실패시킨다(Task 2).
2. **표시 범위가 Kiwi 토큰 경계와 맞지 않는 경우**: 학습 변환에서 제외하고 비율을 기록한다. 평가 정답에서는 제외하지 않는다(Task 5).
3. **관계의 head/tail 표면형이 문단에 두 번 이상 나오는 경우**: 관계는 표면형 기준으로 채점하므로 영향이 없지만, 학습 변환은 첫 등장 기준으로 하고 개수를 기록한다(Task 5).
4. **문서가 모델 입력 길이를 넘는 경우**: 엔티티·관계 예시는 문단 단위로 만들고, 레코드 텍스트가 길면 경고 개수를 기록한다(Task 5).
5. **레코드 필드 값이 레코드 텍스트에 없는 경우**: 검증기가 실패시킨다. 값은 원문 표면형이어야 한다(Task 2).

---

### Task 0: 학습 형식 확인 (스파이크, 버림)

**Files:**
- Create: `/private/tmp/.../scratchpad/spike_joint_training.py` (저장소 밖)

- [ ] **Step 1**: 장난감 예시 6개를 만든다. 한 `InputExample`에 `entities`, `relations=[Relation(...)]`, `structures=[Structure("incident", 관련자=[...], 유형=ChoiceField(value, choices))]`를 함께 넣는다. `TrainingDataset.validate(raise_on_error=True)`를 통과하는지 확인한다.
- [ ] **Step 2**: `max_steps=4`로 LoRA 학습을 돌리고, 학습 후 다음 두 가지가 오류 없이 결과를 내는지 확인한다.
  - `JointIEEngine.extract`
  - `model.extract(text, model.create_schema().structure("incident").field("유형", dtype="str", choices=[...]).field("관련자", dtype="list"))`
- [ ] **Step 3**: 결과를 계획의 ledger에 기록한다. 세 작업을 한 예시에 섞을 수 없으면, Task 5에서 작업별로 예시를 나누도록 판단을 기록한다.

### Task 1: 온톨로지와 로더

**Files:**
- Create: `config/domains/military-security-audit.yaml`, `config/security_audit.py`
- Test: `tests/test_security_audit_ontology.py`

**Interfaces:**
- Produces:
  - `load_ontology(path=DEFAULT_PATH) -> Ontology` (dataclass: `entity_types: dict[str, EntityType]`, `relations: list[RelationType]`, `records: dict[str, RecordType]`)
  - `EntityType(label, pole_type, subtype, description_ko, extract: bool)`
  - `RelationType(type, sources: list[str], targets: list[str], description_ko, extract: bool)`
  - `RecordType(name, fields: list[RecordField])`, `RecordField(name, dtype: "str"|"list", choices: list[str]|None, description_ko)`
  - `Ontology.extract_labels() -> dict[str, str]`: 추출 대상 11개의 레이블과 설명
  - `Ontology.allowed(relation, head_type, tail_type) -> bool`
  - `Ontology.joint_schema(engine)`: JointIE 스키마(추출 대상 관계 10개)
  - `Ontology.record_schema(model, record_name)`: 구조화 스키마

YAML은 create-context-graph 키를 따른다(`inherits`, `domain`, `entity_types[].label/pole_type/subtype/properties`, `relationships[].type/source/target`). 이 실험용 확장 키(`description_ko`, `extract`, `records`)를 추가한다. 엔티티 14개, 관계 17개, 레코드 3개(incident, finding, action)의 내용은 spec의 표를 그대로 옮긴다.

사건 유형 선택지는 6개로 확정한다: 저장매체 무단 반출, 비인가 출입, 비밀문서 관리 소홀, 계정·권한 관리 위반, 보안교육 미이수, 인가 만료자 접근.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_counts():
    o = load_ontology()
    assert len(o.entity_types) == 14 and len(o.extract_labels()) == 11
    assert len(o.relations) == 17 and sum(r.extract for r in o.relations) == 10
    assert set(o.records) == {"incident", "finding", "action"}

def test_allowed():
    o = load_ontology()
    assert o.allowed("HOLDS_POSITION", "Person", "Position")
    assert not o.allowed("HOLDS_POSITION", "Unit", "Position")

def test_incident_type_choices():
    f = {f.name: f for f in load_ontology().records["incident"].fields}
    assert len(f["유형"].choices) == 6 and f["관련자"].dtype == "list"
```

- [ ] **Step 2**: 실행 → ImportError
- [ ] **Step 3**: YAML과 로더 구현
- [ ] **Step 4**: 통과 확인

### Task 2: 표시 형식 파서와 검증기

**Files:**
- Create: `config/annotation.py`
- Test: `tests/test_annotation.py`

**Interfaces:**
- Consumes: `load_ontology`, `Ontology.allowed`
- Produces:
  - 인라인 표시 형식: `[[타입|표면형]]` (예: `[[Person|김OO 상병]]이 [[StorageMedium|USB-17]]을 반출`)
  - `parse_marked(marked: str) -> tuple[str, list[Span]]`: 표시를 지운 원문과 (start, end, 타입)
  - 문서 레코드 JSON (한 줄 = 문서 1개):
    `{"doc_id", "scenario_id", "split", "doc_type", "area_tags", "paragraphs": [marked str], "relations": [{"paragraph": i, "head", "type", "tail"}], "records": [{"record": "incident"|"finding"|"action", "text_paragraphs": [i,...], "fields": {name: str|list}}], "incident_description": [[para, start, end]]?}`
  - `validate_document(doc, ontology) -> list[str]`: 오류 메시지 목록
  - `validate_corpus(docs, ontology) -> list[str]`: 표면형-타입 일관성 포함

- [ ] **Step 1: 실패하는 테스트**

```python
def test_parse_marked():
    text, spans = parse_marked("[[Person|김OO 상병]]이 [[StorageMedium|USB-17]]을 반출")
    assert text == "김OO 상병이 USB-17을 반출"
    assert spans == [(0, 6, "Person"), (8, 14, "StorageMedium")]

def test_relation_must_be_typed_and_present():
    doc = make_doc(paragraphs=["[[Person|김 상병]]이 [[Unit|정보통신대대]] 소속"],
                   relations=[{"paragraph": 0, "head": "김 상병", "type": "HOLDS", "tail": "정보통신대대"}])
    assert any("HOLDS" in e for e in validate_document(doc, load_ontology()))

def test_record_field_value_must_appear_in_text():
    doc = make_doc(paragraphs=["[[Person|김 상병]]이 반출"],
                   records=[{"record": "incident", "text_paragraphs": [0],
                             "fields": {"유형": "저장매체 무단 반출", "관련자": ["이 병장"]}}])
    assert any("이 병장" in e for e in validate_document(doc, load_ontology()))

def test_corpus_type_consistency():
    a = make_doc(paragraphs=["[[Facility|문서보관실]] 점검"])
    b = make_doc(paragraphs=["[[Unit|문서보관실]] 점검"])
    assert any("문서보관실" in e for e in validate_corpus([a, b], load_ontology()))

def test_unknown_type_and_choice_rejected():
    doc = make_doc(paragraphs=["[[Rank|상병]]"], records=[{"record": "incident", "text_paragraphs": [0],
                   "fields": {"유형": "기타"}}])
    errors = validate_document(doc, load_ontology())
    assert any("Rank" in e for e in errors) and any("기타" in e for e in errors)
```

`make_doc`은 테스트 헬퍼이다. 선택형 필드는 선택지 안의 값이어야 하고, 그 외 필드 값은 레코드 텍스트에 표면형 그대로 있어야 한다. 관계의 head/tail은 같은 문단에 표시된 표면형이어야 한다.

- [ ] **Step 2**: 실행 → ImportError
- [ ] **Step 3**: 구현
- [ ] **Step 4**: 통과 확인

### Task 3: 엔티티 사전과 시나리오 생성기

**Files:**
- Create: `experiments/data/security_audit/dictionary.json`, `scripts/security_audit/make_scenarios.py`
- Test: `tests/test_scenarios.py`
- Output: `experiments/data/security_audit/scenarios.jsonl`

**Interfaces:**
- Consumes: `load_ontology`
- Produces:
  - `make_scenarios(n: int, seed: int, dictionary: dict, ontology) -> list[dict]`
  - 시나리오 형식: `{"scenario_id", "split": "train"|"eval", "area": 영역, "incident_type", "entities": [{"id", "type", "surface", "attrs"}], "relations": [{"head": id, "type", "tail": id}], "incident": {...}, "findings": [...], "actions": [...], "doc_types": [...]}`

사전은 가상 이름 60개 이상, 계급, 직책, 부대·부서, 시설·구역, 매체, 문서, 정보체계, 규정, 외부기관 목록을 담는다. 문서 종류는 spec의 8종이다. 시험 규모는 30개(train 24, eval 6)로 정한다. eval 시나리오에만 쓰는 문서 종류는 `보안대책 회의록`으로 한다.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_balanced_and_legal():
    sc = make_scenarios(30, 42, load_dictionary(), load_ontology())
    assert len(sc) == 30 and sum(s["split"] == "eval" for s in sc) == 6
    assert len({s["area"] for s in sc}) == 4 and len({s["incident_type"] for s in sc}) == 6
    o = load_ontology(); types = lambda s: {e["id"]: e["type"] for e in s["entities"]}
    assert all(o.allowed(r["type"], types(s)[r["head"]], types(s)[r["tail"]]) for s in sc for r in s["relations"])

def test_seeded():
    d, o = load_dictionary(), load_ontology()
    assert make_scenarios(30, 42, d, o) == make_scenarios(30, 42, d, o)

def test_meeting_minutes_only_in_eval():
    sc = make_scenarios(30, 42, load_dictionary(), load_ontology())
    assert all("보안대책 회의록" not in s["doc_types"] for s in sc if s["split"] == "train")
```

- [ ] **Step 2**: 실행 → ImportError
- [ ] **Step 3**: 구현하고 `scenarios.jsonl` 생성
- [ ] **Step 4**: 통과 확인

### Task 4: 문서 작성 (서브에이전트) — 시험 30 시나리오

**Files:**
- Create: `scripts/security_audit/writing_prompt.md` (작성 지시), `experiments/data/security_audit/documents.jsonl`

- [ ] **Step 1**: 작성 지시문을 쓴다. 다음 내용을 담는다.
  - 표기 규칙 전문(Global Constraints)과 인라인 표시 형식
  - 문서 레코드 JSON 형식, 문서 종류별 서식(레코드 단위가 드러나는 번호·표·제목)
  - "시나리오에 있는 사실만 쓰고, 쓴 사실만 관계·레코드 정답으로 남긴다"
  - eval 경위서는 `incident_description`을 표시한다
  - 가상 데이터 규칙
- [ ] **Step 2**: 시나리오를 6개씩 나눠 서브에이전트 5개에 맡긴다. 시나리오마다 `doc_types`의 문서를 작성한다(시나리오당 4~6개). eval 시나리오 묶음에는 별도 작성 지시(문체 변주)를 덧붙인다.
- [ ] **Step 3**: `validate_corpus`를 실행한다. 오류가 있는 문서는 오류 메시지와 함께 다시 작성시키고, 오류 0이 될 때까지 반복한다. 기대 결과: 문서 약 120~170개, 검증 오류 0.

### Task 5: 학습·평가 데이터 변환

**Files:**
- Create: `config/security_audit_data.py`
- Test: `tests/test_security_audit_data.py`

**Interfaces:**
- Consumes: `parse_marked`, `load_ontology`, `config.klue.is_aligned`
- Produces:
  - `paragraph_examples(doc, ontology) -> list[ParagraphGold]`
    - `ParagraphGold(text, spans: list[Span], relations: list[tuple[str, str, str]])`
  - `record_examples(doc, ontology) -> list[RecordGold]`
    - `RecordGold(record, text, fields: dict)`
  - `to_training_examples(paragraphs, records, ontology) -> tuple[list[InputExample], dict]`: (예시, 통계: 경계 제외 수, 중복 표면형 수, 긴 레코드 수)
  - `split_docs(docs, dev_ratio=0.15, seed=42) -> dict[str, list]`
    - 키: train, dev, eval
    - dev는 train 시나리오를 시나리오 단위로 나눈 것

- [ ] **Step 1: 실패하는 테스트**

```python
def test_paragraph_gold_from_marked():
    doc = make_doc(paragraphs=["[[Person|이OO 중사]]는 [[Position|보안담당관]]이다"],
                   relations=[{"paragraph": 0, "head": "이OO 중사", "type": "HOLDS_POSITION", "tail": "보안담당관"}])
    [p] = paragraph_examples(doc, load_ontology())
    assert p.spans == [(0, 6, "Person"), (8, 13, "Position")]
    assert p.relations == [("이OO 중사", "HOLDS_POSITION", "보안담당관")]

def test_all_labels_declared_and_misaligned_dropped():
    examples, stats = to_training_examples([ParagraphGold("가나다라", [(1, 3, "Person")], [])], [], load_ontology())
    assert set(examples[0].entities) == set(load_ontology().extract_labels())
    assert stats["misaligned"] == 1 and examples[0].entities["Person"] == []

def test_split_by_scenario():
    s = split_docs([make_doc(scenario_id=f"s{i % 10}", split="train") for i in range(40)]
                   + [make_doc(scenario_id="e1", split="eval")])
    train_sc = {d["scenario_id"] for d in s["train"]}; dev_sc = {d["scenario_id"] for d in s["dev"]}
    assert not (train_sc & dev_sc) and {d["scenario_id"] for d in s["eval"]} == {"e1"}
```

레이블은 GLiNER에 영어 타입명을 그대로 쓰고, 설명은 로더의 한국어 설명을 쓴다. Task 0의 결과에 따라 한 예시에 작업을 섞거나 나눈다.

- [ ] **Step 2**: 실행 → ImportError
- [ ] **Step 3**: 구현
- [ ] **Step 4**: 통과 확인

### Task 6: 채점

**Files:**
- Modify: `config/klue.py` (`score`에 `labels` 인자 추가, 기본값 `LABELS`. 기존 동작 유지)
- Create: `config/security_audit_eval.py`
- Test: `tests/test_security_audit_eval.py`

**Interfaces:**
- Produces:
  - `score_entities(preds: list[set[Span]], golds: list[ParagraphGold], labels) -> dict`: `klue.score` 형식
  - `score_relations(preds: list[set[tuple]], golds) -> dict`: 관계 타입별·전체 P/R/F1
  - `score_records(preds: list[dict], golds: list[RecordGold], ontology) -> dict`: 필드별 F1(목록은 원소 단위, 문자열은 값 일치) + 선택형 필드 정확도
  - `predict_entities(model, golds, labels)`, `predict_relations(engine, golds, ontology)`, `predict_records(model, golds, ontology)`

- [ ] **Step 1: 실패하는 테스트**

```python
def test_relation_triple_f1():
    g = [ParagraphGold("t", [], [("김 상병", "BELONGS_TO", "정보통신대대")])]
    s = score_relations([{("김 상병", "BELONGS_TO", "정보통신대대"), ("김 상병", "HOLDS", "출입증")}], g)
    assert s["overall"]["precision"] == 0.5 and s["overall"]["recall"] == 1.0

def test_record_fields_list_and_choice():
    g = [RecordGold("incident", "t", {"유형": "비인가 출입", "관련자": ["김 상병", "이 병장"]})]
    s = score_records([{"유형": "비인가 출입", "관련자": ["김 상병"]}], g, load_ontology())
    assert s["choice_accuracy"]["유형"] == 1.0
    assert s["fields"]["관련자"]["precision"] == 1.0 and s["fields"]["관련자"]["recall"] == 0.5

def test_klue_score_unchanged_default():
    from config.klue import score
    assert score([{(0, 2, "person")}], [("가나", [(0, 2, "person")])])["overall"]["f1"] == 1.0
```

- [ ] **Step 2**: 실행 → ImportError
- [ ] **Step 3**: 구현
- [ ] **Step 4**: 전체 테스트 통과 확인 (`python -m pytest tests`)

### Task 7: 실행 스크립트와 시험 실행

**Files:**
- Create: `scripts/run_security_audit_sft.py`
- Output: `experiments/results/security_audit_sft_pilot.json`, `outputs/security_audit_lora_pilot/`

**Interfaces:**
- Consumes: Task 1, 2, 5, 6
- Produces: 결과 JSON
  - `run`: 날짜, 체크포인트, 문서·예시 수, 장치
  - `data_stats`
  - `conditions`
    - 키: `zero_shot`, `adapter`
    - 값: `{entities, relations, records, seconds}`. 각 항목은 eval과 dev를 모두 담는다.
  - `by_doc_type`, `by_area`
  - `held_out_doc_type`: 보안대책 회의록만 따로 채점
  - `incident_description`: 방식 A 보조 측정
  - `examples`: 정답 그래프 대 추출 그래프 3~5개

학습은 train으로 한다. 오류 분석과 이후 판단은 dev로 하고, eval은 최종 점수 보고에만 쓴다.

- [ ] **Step 1**: `--limit` 스모크 실행(문서 5개, max_steps 4) → 결과 JSON의 모든 키가 채워져야 한다.
- [ ] **Step 2**: 시험 규모 전체를 백그라운드로 실행 → 종료 코드 0, 결과 JSON 저장.

### Task 8: 노트북, README, 보고

**Files:**
- Create: `experiments/04_security_audit_sft.ipynb`
- Modify: `README.md`

- [ ] **Step 1**: 노트북을 작성한다.
  - 구성: 목적 → 온톨로지(표) → 데이터 생성 방법과 예시 문서(표시 형식, 시나리오 그래프) → 데이터 통계 → zero-shot 대 어댑터 점수(엔티티·관계·레코드, 목표 대비) → 문서 종류·영역별 점수 → 학습에 없던 문서 종류 → 방식 A 보조 측정 → 정답 그래프 대 추출 그래프 → 결론과 한계(시험 규모, 같은 작성자 데이터의 점수 부풀림)
  - 관례: `RUN_EXPERIMENT = False`, 출력 제거, 마크다운 수치는 결과 JSON과 일치
  - 프로젝트 루트에서 모든 셀이 실행되는지, 표의 열 개수가 맞는지 확인한다.
- [ ] **Step 2**: README의 구성 표에 실험 04를 추가하고 실행 방법을 적는다.
- [ ] **Step 3**: 시험 결과를 사용자에게 보고하고, 전체 규모 확장 여부를 묻는다(커밋하지 않음).
