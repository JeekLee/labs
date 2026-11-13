# 보안 감사 온톨로지 SFT v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 공개 서식 기반 Markdown 관리대장과 서술 문서로 가상 데이터를 다시 만들고, 통합 어댑터를 학습해 엔티티·관계·기록 레코드를 정답지 대비 측정한다. 처음 보는 이름과 처음 보는 열 변형은 따로 잰다.

**Architecture:**
- 관리대장은 코드로 렌더링한다. 각 행을 v1과 같은 문서 JSON의 "문단"(대장 정보 + 평탄화한 머리글 + 행, 인라인 표시 포함)으로 만들고, 기록 레코드를 붙인다. 그래서 검증·변환·채점 코드를 그대로 재사용한다.
- 서술 문서는 서브에이전트가 v1 형식으로 쓴다.
- 코드는 `config/security_audit/` 패키지로 정리한다.

**Tech Stack:** Python 3.14, uv, gliner2 2.1.0, kiwipiepy, PyYAML, pytest(`uv run --with pytest python -m pytest`)

**Spec:** `ner/ner-gliner2/docs/2026-11-13-security-audit-sft-v2-design.md` (v1 결정은 `2026-11-13-security-audit-sft-design.md`)

## Global Constraints

- 모든 데이터는 가상이다. 부대 번호는 네 자리 가상 번호(`제9013보병사단` 등)를 쓴다. 개인정보 열(생년월일·주민등록번호·신분증번호)은 쓰지 않는다. 자격 증명 값은 `****`로 쓴다.
- 체크포인트 `fastino/gliner2.5-multi-v1`, `kiwi_splitter`, LoRA(2 epoch, 배치 8, seed 42, `use_lora=True, save_adapter_only=True`)를 쓴다.
- 학습 예시에는 추론과 같은 레이블·필드 설명을 넣는다. 관계에는 설명을 넣지 않는다(v1 리뷰 수정 유지).
- 엔티티는 문자 위치로, 관계와 레코드 값은 문자열로 넘긴다. 정렬 실패 값은 학습에서 빼고 개수를 센다(v1 판단 유지).
- 표기 규칙(spec 「대장의 엔티티 표기 규칙」):
  - Person은 같은 칸에 적힌 계급을 포함한다.
  - StorageMedium은 관리번호, ClassifiedDocument는 건명이다.
  - 역할 필드 값은 Person 범위이다.
- 열로 표현된 지속 관계(BELONGS_TO, HOLDS, RESPONSIBLE_FOR, LOCATED_AT)도 정답에 넣는다.
- 기록 유형은 대장 제목으로 알려진 것으로 본다(행마다 스키마 하나).
- 커밋은 사용자가 요청할 때만 한다. v1은 `9f18cde`에 고정돼 있다.
- 임시 파일은 `outputs/tmp/`(gitignore 대상)에 둔다.

## Review Focus

1. **같은 사실이 서식 변형마다 다른 정답이 되는 경우**: 한 행을 서로 다른 변형으로 렌더링해도 관계·레코드 정답이 같아야 한다(Task 4 테스트).
2. **묶음 칸에서 엔티티 위치가 어긋나는 경우**: `정보통신대대 병장 장지후` 칸에서 Unit과 Person 범위가 정확해야 한다(Task 4 테스트).
3. **빈 칸(승인자 공란 등)**: 레코드에서 그 필드가 빠지고, 검증을 통과해야 한다(Task 4 테스트).
4. **평가 전용 사전·변형이 학습에 새는 경우**: 학습 문서에 평가 전용 표면형이나 변형이 나오면 실패해야 한다(Task 5 테스트).
5. **같은 표면형이 문서마다 다른 타입인 경우**: 학습·평가 사전의 표면형이 서로 겹치지 않고 타입이 하나로 유지돼야 한다(Task 5, `validate_corpus`).

---

### Task 1: 패키지 정리

**Files:**
- Move: `config/security_audit.py` → `config/security_audit/ontology.py`, `config/annotation.py` → `config/security_audit/annotation.py`, `config/security_audit_data.py` → `config/security_audit/data.py`, `config/security_audit_eval.py` → `config/security_audit/evaluate.py`
- Create: `config/security_audit/__init__.py`
- Move: `tests/test_{annotation,scenarios,security_audit_*,review_fixes}.py` → `tests/security_audit/`
- Modify: 위 모듈을 가져오는 import 전부 (`scripts/`, `tests/`)

- [ ] **Step 1**: `git mv`로 옮기고 import를 `config.security_audit.<module>`로 고친다.
- [ ] **Step 2**: `uv run --with pytest python -m pytest tests -q` → 48 passed (v1과 같음)

### Task 2: 온톨로지 v2

**Files:**
- Modify: `config/domains/military-security-audit.yaml`, `config/security_audit/ontology.py`
- Test: `tests/security_audit/test_ontology_v2.py`

**Interfaces:**
- Produces:
  - `RelationType.scopes: tuple[str, ...]`. 값은 `"ledger"`, `"narrative"`.
  - `Ontology.joint_schema(engine, scope: str = "narrative")`: 해당 scope의 추출 관계만 넣는다.
  - `Ontology.records`에 대장 레코드 10종 추가: `document_entry, loan, reading, clearance, custodian, entry, daily_check, medium_entry, medium_transfer, system_entry`. 필드는 spec 표를 따른다. 공통 역할 이름은 spec L2 목록을 그대로 쓴다.

관계 scope는 다음과 같다.
- 지속 관계 BELONGS_TO, HOLDS_POSITION, POSITION_OF, HOLDS, RESPONSIBLE_FOR, LOCATED_AT: `[ledger, narrative]`
- ACCESSED, CARRIED_OUT, STORED_ON, INSPECTED: `[narrative]`

- [ ] **Step 1: 실패하는 테스트**

```python
def test_v2_records_and_scopes():
    o = load_ontology()
    assert {"document_entry", "loan", "reading", "clearance", "custodian", "entry", "daily_check",
            "medium_entry", "medium_transfer", "system_entry", "incident", "finding", "action"} == set(o.records)
    ledger = {r.type for r in o.relations if r.extract and "ledger" in r.scopes}
    assert ledger == {"BELONGS_TO", "HOLDS_POSITION", "POSITION_OF", "HOLDS", "RESPONSIBLE_FOR", "LOCATED_AT"}

def test_shared_role_names_have_one_description():
    o = load_ontology()
    descs = {}
    for r in o.records.values():
        for f in r.fields:
            if f.name in {"승인자", "확인자", "입회자", "보관책임자", "처리자", "관리책임자"}:
                descs.setdefault(f.name, set()).add(f.description_ko)
    assert descs and all(len(v) == 1 for v in descs.values())
```

- [ ] **Step 2**: 실패 확인
- [ ] **Step 3**: YAML과 로더 수정. 기존 v1 테스트의 개수 단언은 v2 값으로 갱신한다.
- [ ] **Step 4**: 전체 테스트 통과

### Task 3: 서식 정의

**Files:**
- Create: `experiments/data/security_audit/forms/<record>.yaml` (대장 10종), `config/security_audit/forms.py`
- Test: `tests/security_audit/test_forms.py`

**Interfaces:**
- Produces:
  - `load_forms(dir) -> dict[str, Form]`
  - `Form(record, title, meta_fields: list[str], columns: list[Column], variants: list[Variant])`
  - `Column(field, kind)`
    - `field`: 레코드 필드 이름. `None`이면 순번처럼 레코드에 없는 열이다.
    - `kind`: `person|unit|date|time|medium_id|medium_kind|doc_title|doc_id|grade|facility|system|choice:<값들>|text|number`
  - `Variant(id, eval_only: bool, headers: dict[field, str], order: list[field], grouped: list[list[field]], date_format: str)` — eval_only 변형의 id는 `<record>_eval`로 끝난다

열 구성은 v1 설계 문서의 「참고 자료」 공개 서식 표를 따른다. 대장마다 변형은 3~4종이고 그중 1종이 `eval_only`이다.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_every_form_has_one_eval_only_variant_and_maps_record_fields():
    forms, o = load_forms(FORMS_DIR), load_ontology()
    assert set(forms) == {"document_entry", "loan", "reading", "clearance", "custodian", "entry",
                          "daily_check", "medium_entry", "medium_transfer", "system_entry"}
    for name, form in forms.items():
        assert sum(v.eval_only for v in form.variants) == 1 and 3 <= len(form.variants) <= 4
        record_fields = {f.name for f in o.records[name].fields}
        assert {c.field for c in form.columns if c.field} | set(form.meta_fields) >= record_fields - {"비고"}

def test_eval_only_variant_differs_in_more_than_names():
    for form in load_forms(FORMS_DIR).values():
        ev = next(v for v in form.variants if v.eval_only)
        others = [v for v in form.variants if not v.eval_only]
        assert all(ev.date_format != o.date_format or ev.grouped != o.grouped or ev.order != o.order for o in others)
```

- [ ] **Step 2–4**: 실패 확인 → 구현 → 통과

### Task 4: 대장 행 렌더링

**Files:**
- Create: `config/security_audit/tables.py`
- Test: `tests/security_audit/test_tables.py`

**Interfaces:**
- Consumes: `Form`, `Variant` (Task 3), `parse_marked` (annotation)
- Produces:
  - `RowValues = dict[str, Value]`, `Value = {"surface": str, "type": str | None}`. `type`은 엔티티 타입이고, 엔티티가 아니면 `None`이다.
  - `render_ledger(form, variant, meta: RowValues, rows: list[RowValues], person_units: dict[str, str]) -> dict`: v1 문서 JSON(`doc_kind="ledger"`, `variant_id`, `paragraphs`, `relations`, `records`)
    - 문단 하나 = 대장 정보 줄 + 머리글 행 + 데이터 행이고, 인라인 표시를 포함한다.
    - 레코드 하나 = 그 문단의 기록 레코드이다.
    - 관계는 열로 표현된 지속 관계(Global Constraints)이다.
  - `format_date(iso: str, fmt: str) -> str`

- [ ] **Step 1: 실패하는 테스트**

```python
def test_same_facts_same_gold_across_variants():
    form = load_forms(FORMS_DIR)["medium_transfer"]
    meta, rows = sample_transfer()          # 테스트 헬퍼: 관리책임자 1, 행 2개(하나는 확인자 공란)
    golds = [render_ledger(form, v, meta, rows, {"장지후 병장": "제9013보병사단 정보통신대대"}) for v in form.variants]
    norm = lambda d: (sorted(map(str, d["relations"])), [r["fields"] for r in d["records"]])
    assert len({json.dumps(norm(g), ensure_ascii=False, sort_keys=True) for g in golds}) == 1

def test_grouped_cell_spans():
    form = load_forms(FORMS_DIR)["entry"]
    v = next(v for v in form.variants if ["소속", "출입자"] in v.grouped)
    doc = render_ledger(form, v, *sample_entry())
    text, spans = parse_marked(doc["paragraphs"][0])
    assert ("병장 장지후", "Person") in {(text[s:e], t) for s, e, t in spans}
    assert ("제9013보병사단 정보통신대대", "Unit") in {(text[s:e], t) for s, e, t in spans}

def test_blank_field_omitted_and_valid():
    form = load_forms(FORMS_DIR)["medium_transfer"]
    doc = render_ledger(form, form.variants[0], *sample_transfer_with_blank_confirm())
    assert "확인자" not in doc["records"][1]["fields"]
    assert validate_document(doc, load_ontology()) == []

def test_format_date():
    assert format_date("2026-04-07", "dot") == "2026. 4. 7." and format_date("2026-04-07", "short") == "26-04-07"
```

머리글이 여러 줄로 병합된 열은 평탄화한 이름을 쓴다(예: `반납 확인관`). 대장 정보 줄은 `제목 / 부대 / 관리책임자: …` 형식이다.

- [ ] **Step 2–4**: 실패 확인 → 구현 → 통과

### Task 5: 사전 분리, 시나리오 v2, 대장 생성

**Files:**
- Create:
  - `experiments/data/security_audit/dictionary/{train,eval,common}.json` (common = 계급, 직책, 비밀등급처럼 나누지 않는 고정 어휘)
  - `scripts/security_audit/render_ledgers.py`
  - 출력: `experiments/data/security_audit/documents/{train,eval}/ledgers.jsonl`
- Modify: `scripts/security_audit/make_scenarios.py` (v2)
- Test: `tests/security_audit/test_scenarios_v2.py`

**Interfaces:**
- Produces:
  - `make_scenarios(n_train=45, n_eval=15, seed=42, dicts, ontology) -> list[dict]`
    - 시나리오에 `ledgers: [{"record", "variant_id", "meta", "rows"}]`가 있다.
    - 사건 유형별 대장 배정: 저장매체 무단 반출 → medium_transfer, medium_entry / 비인가 출입 → entry, daily_check / 비밀문서 관리 소홀 → document_entry, loan, reading / 계정·권한 관리 위반 → system_entry, medium_entry / 보안교육 미이수 → custodian, clearance / 인가 만료자 접근 → clearance, reading, loan
    - 대장당 행 5~8개, 사건 행은 1~2개이다.
  - 평가 시나리오는 엔티티의 약 절반을 eval 사전에서, 대장의 약 절반을 eval_only 변형으로 만든다.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_dictionaries_disjoint():
    tr, ev = load_dict("train"), load_dict("eval")
    for kind in tr:
        assert not set(tr[kind]) & set(ev[kind]), kind

def test_train_has_no_eval_surfaces_or_variants():
    sc = make_scenarios(45, 15, 42, load_dicts(), load_ontology())
    ev_surfaces = {s for k, v in load_dict("eval").items() for s in v}
    forms = load_forms(FORMS_DIR)
    for s in sc:
        if s["split"] == "train":
            assert not any(v.eval_only for l in s["ledgers"] for v in forms[l["record"]].variants
                           if v.id == l["variant_id"])
            assert not any(e["surface"] in ev_surfaces for e in s["entities"])

def test_eval_mixes_seen_and_unseen():
    sc = [s for s in make_scenarios(45, 15, 42, load_dicts(), load_ontology()) if s["split"] == "eval"]
    ledgers = [l for s in sc for l in s["ledgers"]]
    eval_only = sum(l["variant_id"].endswith("eval") for l in ledgers)
    assert 0.3 <= eval_only / len(ledgers) <= 0.7
```

- [ ] **Step 2–4**: 실패 확인 → 구현 → 통과
- [ ] **Step 5**: `render_ledgers.py` 실행 → 대장 문서 생성, `validate_docs.py` → 오류 0

### Task 6: 서술 문서 (서브에이전트)

**Files:**
- Modify: `scripts/security_audit/writing_prompt.md`
  - v2 사전·표기 규칙을 반영한다.
  - 머리글의 점검 대상·작성자 소속도 관계로 표시하게 한다.
  - 선택지 단어를 본문에 쓰지 않는 레코드를 절반 정도 만들게 한다.
- Output: `experiments/data/security_audit/documents/{train,eval}/narrative_*.jsonl`

- [ ] **Step 1**: 시나리오당 보안점검 결과보고서 1개 + 보안사고 경위서 1개를 쓴다. 서브에이전트 6개(학습 4, 평가 2)가 작성한다.
- [ ] **Step 2**: 대장과 서술 문서 전체에 `validate_corpus` → 오류 0. `plain/` 내보내기를 다시 실행한다.

### Task 7: 채점 확장

**Files:**
- Modify: `config/security_audit/evaluate.py`
- Test: `tests/security_audit/test_evaluate_v2.py`

**Interfaces:**
- Produces:
  - `score_by_group(scores_fn, items, key) -> dict[str, dict]`: 임의 그룹 키별로 같은 채점 함수를 적용한다. 키 예시는 `doc_kind`, `variant_seen`, `record`이다.
  - `choice_literal_split(preds, golds, ontology) -> dict`: 선택형 정확도를 정답 단어가 본문에 있을 때와 없을 때로 나눈다.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_choice_literal_split():
    g = [RecordGold("incident", "비인가 출입 확인", {"유형": "비인가 출입"}),
         RecordGold("incident", "무단 진입 확인", {"유형": "비인가 출입"})]
    s = choice_literal_split([{"유형": "비인가 출입"}, {"유형": "보안교육 미이수"}], g, load_ontology())
    assert s["incident.유형"] == {"literal": 1.0, "absent": 0.0, "n_literal": 1, "n_absent": 1}

def test_score_by_group():
    items = [{"k": "a", "x": 1}, {"k": "b", "x": 0}]
    out = score_by_group(lambda xs: {"mean": sum(i["x"] for i in xs) / len(xs)}, items, "k")
    assert out == {"a": {"mean": 1.0}, "b": {"mean": 0.0}}
```

- [ ] **Step 2–4**: 실패 확인 → 구현 → 통과

### Task 8: 실행

**Files:**
- Move/Modify: `scripts/run_security_audit_sft.py` → `scripts/security_audit/run_sft.py`
  - 문서를 `documents/`에서 읽는다.
  - 결과 JSON에 다음 그룹을 추가한다: `by_doc_kind`(ledger/narrative), `by_variant`(seen/unseen, 대장만), `by_record`(대장 레코드별), `entities_by_seen`, `choice_literal_split`
  - 관계 예측은 대장 문단에 `scope="ledger"`, 서술 문단에 `scope="narrative"` 스키마를 쓴다.
- Output: `experiments/results/security_audit_sft.json`, `outputs/security_audit_lora/`

- [ ] **Step 1**: 시나리오 3개(학습 2, 평가 1) 스모크: `--limit-scenarios 3 --max-steps 4` → 모든 키가 채워져야 한다.
- [ ] **Step 2**: 전체 실행을 백그라운드로 돌린다 → 종료 코드 0.

### Task 9: 노트북·README

**Files:**
- Rewrite: `experiments/04_security_audit_sft.ipynb` (v2)
- Modify: `README.md`, `scripts/security_audit/export_plain.py`(새 경로)

- [ ] **Step 1**: 노트북 구성
  - v2 목적과 v1 대비 변경 → 온톨로지(4계층) → 공개 서식과 대장 예시(머리글 변형 비교)
  - 데이터 통계 → 결과 요약(목표 대비)
  - 대장 행 기록 레코드(대장별, 본 변형 / 처음 보는 변형) → 엔티티(본 이름 / 처음 보는 이름) → 관계(대장 / 서술) → 사건 레코드와 선택형(정답 단어 유무)
  - 그래프 예시 → v1 시험 대비 → 결론과 한계
  - 마크다운 수치는 결과 JSON과 일치시키고, 표의 열 개수를 검사한다.
- [ ] **Step 2**: README의 실행 방법과 폴더 구조를 갱신한다.
