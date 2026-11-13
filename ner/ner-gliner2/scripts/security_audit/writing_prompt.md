# 가상 군부대 보안 감사 문서 작성 지시

당신은 **가상의** 군부대 보안 감사 문서를 작성한다. 이 문서는 개체명 인식 모델의 학습·평가 데이터로 쓰인다. 모든 내용은 허구이며, 실존 부대·인물·비밀 내용·실제 보안 절차의 세부를 쓰지 않는다. 보안 규정은 이름과 일반적인 취지만 언급한다.

## 입력

시나리오(JSON) 하나가 주어진다. `entities`(id, type, surface), `relations`(head id, type, tail id), `incident`, `findings`, `actions`, `doc_types`가 들어 있다. `doc_types`에 적힌 문서를 **각각 하나씩** 작성한다.

## 쓰는 원칙

1. 시나리오에 있는 사실만 쓴다. 새 인물·부대·매체·문서·체계·규정을 지어내지 않는다(일반 명사는 자유롭게 쓴다).
2. 문서에 실제로 서술한 사실만 관계와 레코드 정답으로 남긴다. 시나리오의 관계라도 그 문서에 쓰지 않았으면 정답에 넣지 않는다.
3. 문서 종류마다 실제 행정 문서처럼 서식을 갖춘다(제목, 일자, 번호 붙은 항목, 표 형태의 행 등). 한 문서는 문단 3~8개.
4. 문장은 한국어 군 행정 문체로 쓰되, 같은 내용이라도 문서마다 표현을 바꾼다.

## 엔티티 표시

문장 안의 엔티티를 `[[타입|표면형]]`으로 표시한다. 표시하는 타입은 다음 11개뿐이다.

Person, Position, Unit, Organization, Facility, Credential, ClassifiedDocument, StorageMedium, InformationSystem, Regulation, Inspection

- **Person**: 특정 개인. 문서에 적힌 **계급을 범위에 포함**한다. `[[Person|윤지환 하사]]`, `[[Person|하사 윤지환]]`, `[[Person|윤 하사]]` 모두 가능하다. 소속 부대명은 범위에 넣지 않는다.
- **Position**: 직책 자체. `[[Position|정보보호 담당관]]`. `보안담당관 이OO 중사`는 `[[Position|보안담당관]] [[Person|이OO 중사]]`처럼 따로 표시한다.
- **Unit**: 공식 부대·부서명 전체를 한 범위로. `[[Unit|제427기계화여단 군수지원대대]]`. 줄여 쓸 때도 시나리오의 표면형을 그대로 쓴다(줄인 이름을 새로 만들지 않는다).
- **Inspection**: 특정되는 행사명만. `[[Inspection|2026년 1분기 정기 보안점검]]`. 일반 명사 `보안점검`, `보안교육`은 표시하지 않는다.
- 나머지 타입은 시나리오의 표면형을 그대로 쓴다.
- 조사는 범위 밖에 둔다: `[[Person|윤 하사]]가`, `[[StorageMedium|보안USB-100]]을`.
- 이름 없는 직책 언급(`담당관이 확인하였다`)이나 일반 명사(`해당 매체`, `부대`)는 표시하지 않는다.
- 같은 엔티티가 한 문단에 여러 번 나오면 나올 때마다 표시한다.
- 표시는 중첩하지 않는다.

## 출력 형식

문서 하나를 JSON 한 줄로 쓴다(JSONL). 키:

```json
{"doc_id": "<scenario_id>-<번호>", "scenario_id": "...", "split": "train|eval", "doc_type": "...",
 "area_tags": ["<시나리오 area>"],
 "paragraphs": ["표시가 들어간 문단", "..."],
 "relations": [{"paragraph": 0, "head": "표면형", "type": "BELONGS_TO", "tail": "표면형"}],
 "records": [{"record": "incident|finding|action", "text_paragraphs": [0, 1], "fields": {"필드": "값 또는 [값]"}}]}
```

### relations

- 관계 추출 대상 10개만 쓴다: BELONGS_TO(Person→Unit), HOLDS_POSITION(Person→Position), POSITION_OF(Position→Unit), HOLDS(Person→Credential), RESPONSIBLE_FOR(Person→Facility/ClassifiedDocument/StorageMedium/InformationSystem), ACCESSED(Person→Facility/InformationSystem/ClassifiedDocument), CARRIED_OUT(Person→StorageMedium/ClassifiedDocument), STORED_ON(ClassifiedDocument→StorageMedium/InformationSystem), LOCATED_AT(Facility/InformationSystem/StorageMedium→Facility/Unit), INSPECTED(Inspection→Unit/Facility/InformationSystem).
- head와 tail은 **같은 문단에 표시한 표면형 그대로**(그 문단에서 쓴 형태, 예: `윤 하사`). 관계를 서술한 문단 번호를 `paragraph`에 쓴다.

### records

- **incident**: 보안사고 경위서에서 문서 전체(모든 문단)를 레코드 1개로. 필드: `유형`(선택지: 저장매체 무단 반출, 비인가 출입, 비밀문서 관리 소홀, 계정·권한 관리 위반, 보안교육 미이수, 인가 만료자 접근), `관련자`, `매체`, `대상 문서`, `장소`, `위반 규정`(목록), `일자`(문자열).
- **finding**: 보안점검 결과보고서·점검표·점검 결과 문서의 **번호 붙은 발견 항목 하나**가 레코드 1개. `text_paragraphs`는 그 항목의 문단. 필드: `영역`(선택지: 인원보안, 문서보안, 시설보안, 정보통신보안), `대상`, `위반 규정`.
- **action**: 조치 사항을 표의 **행 하나**(한 문단)로 쓰고, 그 행이 레코드 1개. 필드: `조치 유형`(선택지: 재교육, 매체 회수, 권한 회수, 시설 보완, 징계 의뢰, 규정 개정), `담당`, `기한`.
- 선택지 필드 외의 모든 값은 **레코드 문단에 실제로 적힌 문자열 그대로**(표시 기호 제외) 쓴다. 문서에 없는 필드는 빼거나 빈 목록으로 둔다.
- 경위서가 아닌 문서에도 발견 항목이나 조치 행이 있으면 레코드로 남긴다. 해당 서식이 없는 문서(예: 반출입 대장)는 `records`를 빈 목록으로 둔다.

## 검증

작성 후 스스로 확인한다.
- 표시 기호가 짝이 맞는가.
- 관계의 head/tail이 그 문단에 표시돼 있는가, 타입 조합이 위 목록에 맞는가.
- 레코드 값이 레코드 문단에 그대로 있는가.

결과 파일은 지정된 경로에 JSONL로 저장한다. 다른 설명은 출력하지 않는다.
