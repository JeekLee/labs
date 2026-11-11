"""Teacher labeler for the distillation experiment: gpt-5.4-mini tags person/organization/location.

The prompt explains the KLUE-NER annotation rules in prose only. It deliberately contains
no few-shot examples and no KLUE sentences, so no human label leaks into the teacher.
"""

import asyncio
import json
import time
import uuid
from pathlib import Path
from time import perf_counter

from config.klue import LABELS, Span

BACKOFF_SECONDS = 2.0

TEACHER_PROMPT = """한국어 문장에서 개체명을 찾아 사람(person), 기관(organization), 장소(location) 세 타입으로 표시한다.
표시 기준은 한국어 개체명 인식 데이터셋 KLUE-NER의 규칙(한국정보통신기술협회 TTA 개체명 지침 기반)을 따른다.

## 타입 정의
- person: 특정 개인 또는 집단을 가리키는 고유한 이름. 가수 그룹, 밴드, 팀처럼 사람들의 모임을 부르는 고유한 이름도 person이다. 실명뿐 아니라 예명, 별명, 성만 쓴 호칭에서의 성 부분도 고유한 이름이면 포함한다.
- location: 행정구역(국가, 도, 시, 군, 구, 동 등)의 이름이나 산, 강, 바다, 섬 같은 지리적 위치의 이름. 건물이나 시설이 장소로 쓰인 경우도 고유한 이름이면 location이다.
- organization: 기관, 기업, 단체, 정당, 학교, 언론사, 스포츠 구단 등 조직의 고유한 이름.

## 범위 규칙
- 이름에 붙은 조사, 어미, 접미 표현은 범위에서 뺀다. 이름 부분만 표시한다.
- 이름 뒤에 오는 직함, 직책, 호칭, 존칭(대표, 회장, 장관, 의원, 감독, 선수, 씨, 님 등)은 범위에서 뺀다. 이름 앞에 오는 직책이나 소속 설명도 뺀다.
- 이름을 꾸미는 수식어나 일반 명사는 범위에서 뺀다. 단, 공식 이름의 일부인 단어는 포함한다.
- 원문에 적힌 글자를 그대로 쓴다. 띄어쓰기, 표기, 대소문자를 바꾸거나 번역하거나 정규화하지 않는다.
- 같은 이름이 문장에 여러 번 나오면 나온 횟수만큼 각각 표시한다.
- 한 범위는 하나의 타입만 가진다. 서로 겹치는 범위를 동시에 표시하지 않는다.

## 타입 결정
- 같은 이름이라도 문맥에 따라 타입을 정한다. 조직 이름이 장소를 뜻하는 문맥이면 location, 조직의 활동을 뜻하는 문맥이면 organization으로 본다.
- 매체나 상품의 이름이 사물(읽거나 사는 대상)로 쓰인 문맥이면 기관으로 표시하지 않는다.
- 고유한 이름이 아닌 일반 명사구(직업, 역할, 관계를 나타내는 표현, 불특정 다수를 가리키는 표현)는 표시하지 않는다.
- 날짜, 시간, 수량, 금액은 이번 과제의 대상이 아니므로 표시하지 않는다.
- 해당하는 개체명이 없으면 빈 목록을 반환한다.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entities"],
    "properties": {
        "entities": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["text", "type"],
            "properties": {"text": {"type": "string"},
                           "type": {"type": "string", "enum": LABELS}},
        }},
    },
}


def _word_start(text: str, index: int) -> bool:
    return index == 0 or not text[index - 1].isalnum()


def locate_spans(text: str, entities: list[dict]) -> tuple[list[Span], int]:
    """Attach character offsets to the teacher's surfaces.

    Longer surfaces are placed first, an occurrence may not overlap a span already placed,
    and occurrences that start a word win over ones inside a word. Repeated mentions of a
    surface take successive free occurrences.
    """
    spans, dropped, occupied = [], 0, []
    for entity in sorted(entities, key=lambda item: -len(item["text"])):
        surface = entity["text"]
        candidates, index = [], text.find(surface) if surface else -1
        while index >= 0:
            end = index + len(surface)
            if all(end <= s or e <= index for s, e in occupied):
                candidates.append(index)
            index = text.find(surface, index + 1)
        if not candidates:
            dropped += 1
            continue
        start = next((i for i in candidates if _word_start(text, i)), candidates[0])
        occupied.append((start, start + len(surface)))
        spans.append((start, start + len(surface), entity["type"]))
    return sorted(spans), dropped


class TeacherLabeler:
    def __init__(self, model: str, concurrency: int = 16, max_retries: int = 3):
        from openai import AsyncOpenAI

        self.model = model
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.client = AsyncOpenAI(max_retries=0)  # retries are counted here, not hidden in the SDK

    async def _request(self, text: str) -> tuple[str, int, int]:
        response = await self.client.responses.create(
            model=self.model,
            instructions=TEACHER_PROMPT,
            input=text,
            text={"format": {"type": "json_schema", "name": "entities",
                             "schema": RESPONSE_SCHEMA, "strict": True}},
        )
        return response.output_text, response.usage.input_tokens, response.usage.output_tokens

    async def _label_one(self, text: str, gate: asyncio.Semaphore, run_id: str) -> dict:
        async with gate:
            started_at, started = time.time(), perf_counter()
            entities, error, attempts, input_tokens, output_tokens = [], None, 0, 0, 0
            for attempt in range(self.max_retries):
                attempts += 1
                try:
                    output, used_in, used_out = await self._request(text)
                    input_tokens += used_in
                    output_tokens += used_out
                    entities, error = json.loads(output)["entities"], None
                    break
                except Exception as exc:  # recorded per sentence; the run continues
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt + 1 < self.max_retries:
                        await asyncio.sleep(BACKOFF_SECONDS * 2 ** attempt)
            spans, dropped = locate_spans(text, entities)
            return {"text": text, "entities": entities, "spans": [list(s) for s in spans],
                    "dropped": dropped, "input_tokens": input_tokens, "output_tokens": output_tokens,
                    "attempts": attempts, "seconds": perf_counter() - started,
                    "started_at": started_at, "finished_at": time.time(), "run_id": run_id,
                    "error": error}

    async def label_many(self, texts: list[str], cache_path: Path) -> list[dict]:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        done = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                record = json.loads(line)
                done[record["text"]] = record
        gate = asyncio.Semaphore(self.concurrency)
        todo = list(dict.fromkeys(t for t in texts if t not in done))
        with cache_path.open("a") as cache:
            run_id = uuid.uuid4().hex
            for future in asyncio.as_completed([self._label_one(t, gate, run_id) for t in todo]):
                record = await future
                cache.write(json.dumps(record, ensure_ascii=False) + "\n")
                cache.flush()
                done[record["text"]] = record
        return [done[t] for t in texts]
