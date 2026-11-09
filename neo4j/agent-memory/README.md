# Neo4j Agent Memory Lab

`neo4j-agent-memory`를 사용하여 대화 저장, 엔티티 추출, 컨텍스트 검색과 그래프 시각화를 실험하는 노트북 모음이다.

## 구성

| 파일 | 내용 |
|---|---|
| [개념과 동작 구조](notebooks/00_agent_memory_overview.ipynb) | 기억 계층, 지식 추출, 검색 범위와 클라이언트 생명주기를 설명한다. |
| [단기 기억 실험](notebooks/01_short_term_memory.ipynb) | 대화 저장, 그래프 시각화, 이력 조회, 컨텍스트 검색과 세션 정리를 실험한다. |
| [장기 기억: 엔티티와 관계](notebooks/02_long_term_memory_entity.ipynb) | 엔티티·관계 저장, 그래프 시각화와 데이터 정리를 실험한다. |
| [장기 기억: Fact](notebooks/03_long_term_memory_fact.ipynb) | 일반·유효기간 Fact 저장, 시각화와 데이터 정리를 실험한다. |
| [실험: Hybrid와 LLM-only 추출](experiments/01_entity_extraction_hybrid_vs_llm.ipynb) | 노트북 01 메시지에서 GLiNER+LLM과 LLM 단독 추출의 결과·해소·관계 연결을 비교한다. |
| [실험: 언어와 GLiNER 체크포인트](experiments/02_entity_extraction_by_language.ipynb) | 한국어·영어 문장 쌍으로 GLiNER base/multi와 LLM 추출을 비교한다. |
| [메모리 설정](config/memory_client.py) | 환경변수와 추출·모델 설정으로 `MemorySettings`를 구성한다. |
| [공유 유틸리티](../neo4j-utils/) | 여러 Neo4j 실험에서 사용하는 그래프 시각화 함수를 제공한다. |

## 실행 환경

- Python 3.14 이상 및 `uv`
- 접속 가능한 Neo4j 인스턴스
- OpenAI API 키
- Jupyter 노트북을 지원하는 편집기

## 실행 방법

프로젝트 루트에서 의존성을 설치하고 환경변수 파일을 준비한다.

```bash
uv sync --locked
cp .env.example .env
```

공유 패키지 `neo4j-utils`는 `../neo4j-utils`의 소스를 editable 의존성으로 사용한다. `labs` 저장소를 함께 체크아웃한 구조에서 실행한다.

`.env`에 Neo4j 접속 정보와 OpenAI API 키를 입력한다. 실제 값은 로컬 `.env`에서만 관리하며, 저장소에는 자리표시자를 포함한 `.env.example`만 게시한다.

편집기에서 프로젝트 루트를 열고 `.venv`의 Python을 노트북 커널로 선택한다. `00` 노트북으로 개념을 확인한 뒤 `01`~`03`에서 실험 주제를 선택하고 해당 노트북의 셀을 순서대로 실행한다. 각 실험 노트북은 자체 클라이언트 연결·종료 단계를 포함하므로 별도의 커널에서 독립적으로 실행할 수 있다. 프로젝트 루트가 Python 모듈 검색 경로에 포함되어야 `config.memory_client`를 가져올 수 있다.

현재 설정은 GLiNER와 LLM을 이용한 엔티티·관계 추출 및 OpenAI 임베딩을 사용한다. 모델 설정은 `config/memory_client.py`에서 변경한다.

메시지 저장 셀을 반복 실행하면 메시지가 추가된다. 세션 삭제는 해당 대화와 메시지를 삭제하며, 추출된 엔티티 등 장기 기억은 남을 수 있다. 실험 종료 시 클라이언트 종료 셀을 실행한다.

노트북 실행 결과와 위젯 상태는 공개 파일에서 제거되어 있다. 그래프는 시각화 셀을 실행하여 확인한다.

## 참고 자료

- [Neo4j Agent Memory](https://neo4j.com/labs/agent-memory/)
- [Neo4j Python Graph Visualization](https://neo4j.com/docs/python-graph-visualization/current/)
