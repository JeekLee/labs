# Neo4j Utilities

여러 Neo4j 실험 프로젝트에서 재사용하는 Python 라이브러리이다. Python 3.11 이상을 지원하며, 현재는 노트북 그래프 시각화 함수를 제공한다.

## 설치

`agent-memory`처럼 `neo4j-utils`와 같은 상위 디렉터리에 있는 프로젝트에서 다음 명령을 실행한다.

```bash
uv add --editable ../neo4j-utils
```

설치 이름은 `neo4j-utils`, Python 모듈 이름은 `graph_visualizer`이다. 각 실험은 자체 가상환경과 잠금 파일을 유지한다. Editable 설치를 통해 이 디렉터리의 소스를 참조하므로, 다른 환경에서 실행할 때도 공유 패키지 디렉터리를 함께 준비해야 한다.

## 그래프 시각화

`visualize(session, query, parameters=None, *, height="600px")`는 공식 Neo4j 드라이버의 `AsyncSession`으로 쿼리를 실행하고 `neo4j-viz` 위젯을 표시한다. 쿼리는 노드·관계·경로를 반환해야 한다.

```python
from typing import LiteralString

from graph_visualizer import visualize

query: LiteralString = """
MATCH path = (n:Entity {id: $entity_id})-[*1..2]-(m)
RETURN path
LIMIT 100
"""

# driver는 호출자가 생성한 neo4j.AsyncDriver 객체이다.
async with driver.session(database="neo4j") as session:
    await visualize(session, query, {"entity_id": entity_id})
```

`neo4j-agent-memory`의 기존 연결을 재사용할 때는 다음과 같이 호출한다.

```python
async with memory_client.graph.session() as session:
    await visualize(session, query, {"entity_id": str(entity.id)})
```

세션과 클라이언트의 생성·종료는 호출자가 관리한다. 함수는 전역 클라이언트나 특정 `session_id`를 참조하지 않는다.

Fact 노드의 캡션은 `subject — predicate → object`로 표시한다. 그 외 노드 캡션은 `name`, `content`, `session_id`, 기본 캡션 순서로 선택한다. 캡션은 최대 80자로 표시하며 원문과 속성은 유지한다. 관계는 저장된 구체적 유형이 있으면 해당 값을 표시한다. 조회 결과에 노드가 없으면 안내문을 출력한다.
