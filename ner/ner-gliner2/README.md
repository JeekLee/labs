# GLiNER2 Lab

[GLiNER2](https://github.com/fastino-ai/GLiNER2)(GLiNER2.5 체크포인트)로 한국어 엔티티·관계 추출을 실험하고, 같은 스키마의 LLM 추출과 비교하는 노트북 모음이다.

## 구성

| 파일 | 내용 |
|---|---|
| [언어와 체크포인트](experiments/01_language_and_checkpoint.ipynb) | 한국어·영어 문장 쌍으로 GLiNER base/multi 체크포인트와 LLM 추출을 비교하고, 조사·표기 프로브로 오류 원인을 분리한다. |
| [단어 분할기](experiments/02_korean_word_splitter.ipynb) | GLiNER 단어 분할기(공백·문자·형태소)별 한국어 엔티티·관계 추출을 참고 정답과 LLM 기준으로 비교한다. |
| [스키마](config/ontology.py) | 두 추출기가 공유하는 POLE+O 엔티티·관계 타입을 정의한다. |
| [추출기](config/extractors.py) | GLiNER2 JointIE 추출기, OpenAI 구조화 출력 추출기, Kiwi 형태소 분할기를 제공한다. |

실험 결과는 `experiments/results/`의 JSON 파일에 저장되며, 노트북의 마크다운 셀에 실행 결과의 수치와 해석을 함께 기록한다.

## 실행 환경

- Python 3.14 이상 및 `uv`
- OpenAI API 키 (LLM 기준선에만 필요)
- Jupyter 노트북을 지원하는 편집기

## 실행 방법

프로젝트 루트에서 의존성을 설치하고 환경변수 파일을 준비한다.

```bash
uv sync --locked
cp .env.example .env
```

`.env`에 OpenAI API 키와 LLM 모델 이름을 입력한다. 편집기에서 프로젝트 루트를 열고 `.venv`의 Python을 노트북 커널로 선택한다.

각 노트북은 기본적으로 저장된 결과 파일을 불러와 분석만 수행한다(`RUN_EXPERIMENT = False`). `True`로 바꾸면 GLiNER 체크포인트를 Hugging Face에서 내려받고 LLM API를 호출하여 결과 파일을 덮어쓴다. GLiNER는 CPU에서 실행한다.

## 배경

이 실험은 [Neo4j Agent Memory Lab](../../neo4j/agent-memory/)에서 SDK의 GLiNER+LLM 추출 파이프라인이 한국어 대화에서 `GCP에서`, `오라클(PERSON)` 같은 엔티티를 저장하는 현상을 관찰한 데서 시작했다. 여기서는 SDK 없이 `gliner2`를 직접 호출하여 원인을 분석한다. SDK 통합 관점의 결과(병합·해소·관계 연결)는 agent-memory의 [Hybrid와 LLM-only 비교](../../neo4j/agent-memory/experiments/01_entity_extraction_hybrid_vs_llm.ipynb)에 있다.

## 참고 자료

- [GLiNER2](https://github.com/fastino-ai/GLiNER2)
- [fastino/gliner2.5-base-v1](https://huggingface.co/fastino/gliner2.5-base-v1), [fastino/gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1)
- [Introducing GLiNER2.5](https://fastino.ai/blog/gliner2-5-span-free-information-extraction)
- [Kiwi 한국어 형태소 분석기](https://github.com/bab2min/kiwipiepy)
- [taeminlee/gliner_ko](https://huggingface.co/taeminlee/gliner_ko): mecab 형태소 분석을 사용하는 한국어 GLiNER(v1) 모델
