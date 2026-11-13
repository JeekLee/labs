# 가상 군부대 보안 감사 서술 문서 작성 지시 (v2)

당신은 **가상의** 군부대 보안 감사 문서를 작성한다. 이 문서는 개체명·관계 추출 모델의 학습·평가 데이터로 쓰인다. 모든 내용은 허구이며, 실존 부대·인물·비밀 내용·실제 보안 절차의 세부를 쓰지 않는다. 비밀번호·PIN 같은 자격 증명 값은 쓰지 않는다(필요하면 `****`).

## 입력

시나리오(JSON) 하나가 주어진다. 주요 키:
- `unit`(부대), `actor`(사고 관련자), `officer`(담당 간부), `officer_position`(그 직책), `incident_type`(사건 유형)
- `incident`(유형, 관련자, 매체, 대상 문서, 장소, 위반 규정, 일자)
- `findings`(영역, 대상, 위반 규정), `actions`(조치 유형, 담당, 기한)
- `entities`: 이 시나리오에 등장하는 모든 고유 이름(surface, type)
- `ledgers`: 같은 시나리오의 관리대장(참고용)

시나리오마다 다음 두 문서를 **각각 하나씩** 작성한다.
1. `보안사고 경위서`: 사고 개요, 관련자, 경위, 위반 규정, 조치를 항목별로 서술한다(문단 4~7개).
2. `보안점검 결과보고서`: 머리글(점검명, 점검 대상 부대, 점검자), 번호 붙은 발견 사항, 조치 사항 표를 쓴다(문단 4~8개). 조치 사항은 표의 행 하나를 한 문단으로 쓴다.

## 쓰는 원칙

1. 시나리오에 있는 사실과 `entities`의 이름만 쓴다. 새 인물·부대·매체·문서·체계·규정을 지어내지 않는다.
2. 점검명은 `2026년 N분기 정기 보안점검`처럼 쓰고 Inspection으로 표시한다(N은 incident 일자의 분기).
3. 문서에 실제로 서술한 사실만 관계와 레코드 정답으로 남긴다.
4. 같은 내용이라도 문서마다 표현을 바꾼다. 한국어 군 행정 문체로 쓴다.
5. **선택형 정답 단어를 절반쯤은 본문에 쓰지 않는다.** 시나리오 두 개 중 하나꼴로 사건 유형(예: `저장매체 무단 반출`)이나 보안 영역(예: `정보통신보안`)을 그 단어 그대로 쓰지 말고, 사실만 서술한다(예: "승인 없이 USB를 부대 밖으로 가지고 나감"). 나머지 절반은 그 단어를 써도 된다.

## 엔티티 표시

문장 안의 엔티티를 `[[타입|표면형]]`으로 표시한다. 표시하는 타입은 다음 11개뿐이다.

Person, Position, Unit, Organization, Facility, Credential, ClassifiedDocument, StorageMedium, InformationSystem, Regulation, Inspection

- **표면형은 `entities`에 있는 그대로 쓴다.** 사람은 계급까지 포함된 형태(`장지후 병장` 또는 `병장 장지후`)를 그대로 쓴다. 두 번째 언급부터는 성+계급 줄임(`장 병장`)도 쓸 수 있고, 그것도 Person으로 표시한다.
- 직책은 Position으로 따로 표시한다: `[[Position|보안담당관]] [[Person|이준서 중사]]`.
- 조사는 범위 밖에 둔다: `[[Person|장지후 병장]]이`.
- 이름 없는 직책 언급이나 일반 명사(`해당 매체`, `부대`)는 표시하지 않는다.
- 같은 엔티티가 한 문단에 여러 번 나오면 나올 때마다 표시한다. 표시는 중첩하지 않는다.

## 출력 형식

문서 하나를 JSON 한 줄로 쓴다(JSONL). 키:

```json
{"doc_id": "<scenario_id>-N<번호>", "scenario_id": "...", "split": "train|eval", "doc_type": "보안사고 경위서|보안점검 결과보고서",
 "doc_kind": "narrative", "area_tags": ["<시나리오 area>"],
 "paragraphs": ["표시가 들어간 문단", "..."],
 "relations": [{"paragraph": 0, "head": "표면형", "type": "BELONGS_TO", "tail": "표면형"}],
 "records": [{"record": "incident|finding|action", "text_paragraphs": [0, 1], "fields": {"필드": "값 또는 [값]"}}]}
```

### relations

- 쓸 수 있는 관계는 10개이다: BELONGS_TO(Person→Unit), HOLDS_POSITION(Person→Position), POSITION_OF(Position→Unit), HOLDS(Person→Credential), RESPONSIBLE_FOR(Person→Facility/ClassifiedDocument/StorageMedium/InformationSystem), ACCESSED(Person→Facility/InformationSystem/ClassifiedDocument), CARRIED_OUT(Person→StorageMedium/ClassifiedDocument), STORED_ON(ClassifiedDocument→StorageMedium/InformationSystem), LOCATED_AT(Facility/InformationSystem/StorageMedium→Facility/Unit), INSPECTED(Inspection→Unit/Facility/InformationSystem).
- head와 tail은 **같은 문단에 표시한 표면형 그대로** 쓴다.
- **문서에서 확인할 수 있는 관계는 빠짐없이 표시한다.** 머리글의 `점검 대상: <부대>`는 INSPECTED, `작성: <부대> <직책> <사람>`은 BELONGS_TO, HOLDS_POSITION, POSITION_OF로 표시한다.
- 행위가 다른 관계를 한 쌍에 함께 붙이지 않는다. 열람만 했으면 ACCESSED만 쓰고, 반출한 경우에만 CARRIED_OUT을 쓴다. 관리 책임이 명시된 경우에만 RESPONSIBLE_FOR를 쓴다.

### records

- **incident**: 보안사고 경위서 전체(모든 문단)를 레코드 1개로 쓴다. 필드는 `유형`(선택지: 저장매체 무단 반출, 비인가 출입, 비밀문서 관리 소홀, 계정·권한 관리 위반, 보안교육 미이수, 인가 만료자 접근), `관련자`, `매체`, `대상 문서`, `장소`, `위반 규정`(목록), `일자`(문자열, 본문에 쓴 형식 그대로)이다.
- **finding**: 결과보고서의 번호 붙은 발견 항목 하나가 레코드 1개이다. 필드는 `영역`(선택지: 인원보안, 문서보안, 시설보안, 정보통신보안), `대상`, `위반 규정`이다.
- **action**: 조치 사항 표의 행 하나(한 문단)가 레코드 1개이다. 필드는 `조치 유형`(선택지: 재교육, 매체 회수, 권한 회수, 시설 보완, 징계 의뢰, 규정 개정), `담당`, `기한`이다.
- 선택지 필드 외의 값은 **레코드 문단에 실제로 적힌 문자열 그대로**(표시 기호 제외) 쓴다. 문서에 없는 필드는 뺀다.

## 검증

작성 후 프로젝트 루트에서 다음을 실행해 오류가 0이 될 때까지 고친다.

```bash
env -u VIRTUAL_ENV uv run python scripts/security_audit/validate_docs.py <내 출력 파일>
```

결과 파일은 지정된 경로에 JSONL로 저장한다. 다른 파일은 수정하지 않는다.
