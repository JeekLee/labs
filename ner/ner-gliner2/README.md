# GLiNER2 Lab

[GLiNER2](https://github.com/fastino-ai/GLiNER2)(GLiNER2.5 체크포인트)의 개념과 사용법을 정리하고, 한국어 엔티티·관계 추출을 실험·학습하여 같은 스키마의 LLM 추출과 비교하는 노트북 모음이다.

## 구성

| 파일 | 내용 |
|---|---|
| [개념과 동작 구조](notebooks/00_gliner2_overview.ipynb) | 추출 방식, 아키텍처 계보(GLiNER → GLiNER2 → GLiNER2.5), 스키마, 단어 분할, threshold, JointIE, 학습을 설명한다. |
| [기본 사용법](notebooks/01_basic_usage.ipynb) | 한국어 문장으로 엔티티·분류·구조화 레코드·관계 추출과 JointIE를 실행한다. |
| [한국어 미세조정](notebooks/02_korean_fine_tuning.ipynb) | KLUE-NER로 LoRA 어댑터를 학습하고 학습 전후 성능을 비교한다. |
| [언어와 체크포인트](experiments/01_language_and_checkpoint.ipynb) | 한국어·영어 문장 쌍으로 GLiNER base/multi 체크포인트와 LLM 추출을 비교하고, 조사·표기 프로브로 오류 원인을 분리한다. |
| [단어 분할기](experiments/02_korean_word_splitter.ipynb) | GLiNER 단어 분할기(공백·문자·형태소)별 한국어 엔티티·관계 추출을 참고 정답과 LLM 기준으로 비교한다. |
| [LLM 증류](experiments/03_llm_distillation.ipynb) | 같은 KLUE 문장에 `gpt-5.4-mini`가 단 정답으로 LoRA 어댑터를 학습하고, 사람 정답으로 학습한 어댑터와 성능·비용을 비교한다. |
| [스키마](config/ontology.py) | 두 추출기가 공유하는 POLE+O 엔티티·관계 타입을 정의한다. |
| [추출기](config/extractors.py) | GLiNER2 JointIE 추출기, OpenAI 구조화 출력 추출기, Kiwi 형태소 분할기를 제공한다. |
| [KLUE 데이터](config/klue.py), [선생님 라벨러](config/teacher.py) | KLUE-NER 변환·정확 일치 채점·오류 분류와, 증류 실험의 `gpt-5.4-mini` 라벨러를 제공한다. |

`notebooks/`는 개념과 사용법을 순서대로 따라가는 설명형 노트북이고, `experiments/`는 조건을 통제한 비교 실험이다. 실험 결과는 `experiments/results/`의 JSON 파일에 저장되며, 두 폴더 모두 마크다운 셀에 실행 결과의 수치와 해석을 함께 기록한다.

## 실행 환경

- Python 3.14 이상 및 `uv`
- OpenAI API 키 (실험 노트북의 LLM 기준선에만 필요)
- Jupyter 노트북을 지원하는 편집기

## 실행 방법

프로젝트 루트에서 의존성을 설치하고 환경변수 파일을 준비한다.

```bash
uv sync --locked
cp .env.example .env
```

`.env`에 OpenAI API 키와 LLM 모델 이름을 입력한다. 편집기에서 프로젝트 루트를 열고 `.venv`의 Python을 노트북 커널로 선택한다. `notebooks/`의 노트북은 프로젝트 루트가 Python 모듈 검색 경로에 포함되어야 `config.extractors`를 가져올 수 있고, 학습 결과도 프로젝트 루트 기준 경로(`outputs/`)에 저장한다. 노트북의 작업 디렉터리를 프로젝트 루트로 설정한다(예: VS Code의 `jupyter.notebookFileRoot`를 `${workspaceFolder}`로 지정).

`notebooks/`는 `00` → `01` → `02` 순서로 읽는다. `01`과 `02`는 첫 실행 시 GLiNER 체크포인트를, `02`는 KLUE-NER 데이터를 Hugging Face에서 내려받는다. `02`는 학습과 평가를 포함해 CPU 기준 약 10분이 걸리며, 학습한 어댑터는 `outputs/`에 저장되고 저장소에는 포함하지 않는다.

`experiments/`의 노트북은 기본적으로 저장된 결과 파일을 불러와 분석만 수행한다(`RUN_EXPERIMENT = False`). `True`로 바꾸면 GLiNER 체크포인트를 내려받고 LLM API를 호출하여 결과 파일을 덮어쓴다. 모든 GLiNER 연산은 CPU에서 실행한다.

`03`의 실험은 노트북이 아니라 스크립트로 실행한다. `notebooks/02`에서 학습한 KLUE 어댑터(`outputs/klue_ner_lora/final`)가 먼저 있어야 하며, 라벨링 API 비용은 약 1.3달러, 전체 실행은 CPU 기준 약 20분이다. 단계별 결과는 `outputs/distill/`에 저장되어 중단 후 이어서 실행할 수 있다.

```bash
uv run python scripts/run_llm_distillation.py --price-input 0.75 --price-output 4.5 \
    --price-source https://developers.openai.com/api/docs/pricing
```

순수 로직(KLUE 변환·채점, 선생님 라벨러)의 테스트는 `uv run --with pytest python -m pytest tests`로 실행한다.

## 배경

이 실험은 [Neo4j Agent Memory Lab](../../neo4j/agent-memory/)에서 SDK의 GLiNER+LLM 추출 파이프라인이 한국어 대화에서 `GCP에서`, `오라클(PERSON)` 같은 엔티티를 저장하는 현상을 관찰한 데서 시작했다. 여기서는 SDK 없이 `gliner2`를 직접 호출하여 원인을 분석한다. SDK 통합 관점의 결과(병합·해소·관계 연결)는 agent-memory의 [Hybrid와 LLM-only 비교](../../neo4j/agent-memory/experiments/01_entity_extraction_hybrid_vs_llm.ipynb)에 있다.

## 참고 자료

- [GLiNER2](https://github.com/fastino-ai/GLiNER2)
- [fastino/gliner2.5-base-v1](https://huggingface.co/fastino/gliner2.5-base-v1), [fastino/gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1)
- [Introducing GLiNER2.5](https://fastino.ai/blog/gliner2-5-span-free-information-extraction)
- [Kiwi 한국어 형태소 분석기](https://github.com/bab2min/kiwipiepy)
- [KLUE: Korean Language Understanding Evaluation](https://klue-benchmark.com/) (KLUE-NER, CC BY-SA 4.0)
- [taeminlee/gliner_ko](https://huggingface.co/taeminlee/gliner_ko): mecab 형태소 분석을 사용하는 한국어 GLiNER(v1) 모델
