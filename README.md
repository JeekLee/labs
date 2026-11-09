# Labs

기술 개념과 라이브러리의 동작을 확인하기 위한 실험 저장소이다. 주제별 디렉터리에 코드와 노트북, 실행 환경을 구성한다.

## 실험 목록

| 디렉터리 | 내용 |
|---|---|
| [neo4j/agent-memory](neo4j/agent-memory/) | Neo4j 기반 에이전트 메모리의 대화 저장, 지식 추출, 컨텍스트 검색과 그래프 시각화를 실험한다. |
| [neo4j/neo4j-utils](neo4j/neo4j-utils/) | 여러 Neo4j 실험에서 재사용하는 Python 유틸리티 패키지이다. |

## 실행

각 실험 디렉터리의 README에 따라 환경을 구성한다. Neo4j Agent Memory 실험은 다음 위치에서 시작한다.

```bash
cd neo4j/agent-memory
uv sync --locked
cp .env.example .env
```

실제 접속 정보와 API 키는 해당 실험의 로컬 `.env`에서 관리한다. 저장소에는 `.env.example`만 포함하며, 가상환경·캐시·IDE 설정·노트북 실행 결과는 공개 대상에서 제외한다.
