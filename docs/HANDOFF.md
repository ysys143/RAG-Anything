# RAG-Anything EZIS 프로젝트 Handoff 문서

## 프로젝트 개요

위데이터랩의 EZIS 제품 매뉴얼을 RAG 시스템으로 구축하여 고객 지원 및 제품 안내에 활용하는 프로젝트.

- **기반 프레임워크**: RAGAnything (LightRAG 확장)
- **백엔드**: PostgreSQL (pgvector + Apache AGE)
- **LLM**: OpenAI GPT-4.1 (텍스트), GPT-4.1 (VLM)

## 디렉토리 구조

```
RAG-Anything-1/
├── scripts/
│   ├── ingest.py          # 문서 인덱싱
│   ├── query.py           # RAG 질의 (CLI)
│   └── cleanup.py         # DB 정리
├── upstream/
│   └── raganything/       # RAGAnything 소스 (수정됨)
│       ├── query.py       # 쿼리 로직 (VLM 스트리밍, 이미지 참조 반환)
│       ├── prompt.py      # 프롬프트 템플릿 (JSON 포맷팅 지침 추가)
│       └── modalprocessors.py  # 멀티모달 처리
├── docker/
│   └── docker-compose.yml # PostgreSQL 컨테이너
├── manuals/               # 원본 PDF 문서
├── output/                # 파싱된 결과 (이미지, 마크다운)
└── rag_storage/           # (미사용, PG 백엔드 사용)
```

## 핵심 스크립트

### 1. 문서 인덱싱 (`scripts/ingest.py`)

```bash
# 단일 파일
uv run python scripts/ingest.py manuals/EZIS_Oracle_Manual_20251216.pdf

# 폴더 전체
uv run python scripts/ingest.py manuals/

# 강제 재처리
uv run python scripts/ingest.py manuals/ --force
```

### 2. RAG 질의 (`scripts/query.py`)

```bash
# 단일 질의
uv run python scripts/query.py "EZIS 시스템 소개"

# 스트리밍 응답
uv run python scripts/query.py "Oracle 연결 설정" --stream

# 참조 이미지 표시
uv run python scripts/query.py "화면 구성 설명" --show-images

# 대화형 모드
uv run python scripts/query.py --interactive

# 옵션 조합
uv run python scripts/query.py -i --stream --show-images
```

**CLI 옵션:**
| 옵션 | 설명 |
|------|------|
| `--mode` | local, global, hybrid, naive, mix (기본: mix) |
| `--stream` | 스트리밍 응답 |
| `--show-images` | 참조 이미지 경로 표시 |
| `--no-vlm` | VLM 비활성화 |
| `--system-prompt` | 커스텀 페르소나 |
| `--interactive` | 대화형 모드 |

### 3. DB 정리 (`scripts/cleanup.py`)

```bash
uv run python scripts/cleanup.py
```

## 최근 변경 사항 (2025-12-30)

### 1. VLM 스트리밍 지원
- `aquery_vlm_enhanced`에서 `stream=True` 지원
- OpenAI Vision API 스트리밍 응답 처리

### 2. 참조 이미지 반환 기능
- VLM 프롬프트에 참조 이미지 명시 요청 추가
- `[REFERENCED_IMAGES: 1, 3]` 태그 파싱
- `return_images=True` 시 dict 반환: `{"response": str, "referenced_images": list[str]}`

### 3. Graceful Shutdown
- SIGINT/SIGTERM 시그널 핸들링
- Pending task 정리로 "Task was destroyed" 경고 제거

### 4. 프롬프트 튜닝
- JSON 포맷팅 지침 추가 (LaTeX 이스케이프 등)
- 위데이터랩 DBA 페르소나 기본 설정

### 5. 디버그 로깅 정리
- 이미지 처리 verbose 로그 제거
- regex fallback warning → debug 변경

## 환경 설정

### 필수 환경 변수 (`.env`)

```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4.1-mini
VISION_MODEL=gpt-4.1
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=pgvector
POSTGRES_PASSWORD=pgvector
POSTGRES_DATABASE=ezis_rag
```

### Docker 실행

```bash
cd docker
docker compose up -d
```

## 데이터베이스 스키마

PostgreSQL 테이블:
- `lightrag_doc_chunks`: 문서 청크
- `lightrag_doc_full`: 전체 문서
- `lightrag_doc_status`: 처리 상태
- `lightrag_full_entities`: 추출된 엔티티
- `lightrag_full_relations`: 엔티티 간 관계
- `lightrag_vdb_*`: 벡터 임베딩
- `lightrag_llm_cache`: LLM 응답 캐시

## API 사용법 (Python)

```python
from raganything import RAGAnything, RAGAnythingConfig

# 초기화
rag = RAGAnything(
    config=RAGAnythingConfig(working_dir="./rag_storage"),
    llm_model_func=llm_func,
    vision_model_func=vision_func,
    embedding_func=embed_func,
    lightrag_kwargs={
        "kv_storage": "PGKVStorage",
        "vector_storage": "PGVectorStorage",
        "graph_storage": "PGGraphStorage",
        "doc_status_storage": "PGDocStatusStorage",
    },
)

# 기본 쿼리
result = await rag.aquery("질문", mode="mix")

# 스트리밍
async for chunk in await rag.aquery("질문", stream=True):
    print(chunk, end="")

# 참조 이미지 포함
result = await rag.aquery("질문", return_images=True)
print(result["response"])
print(result["referenced_images"])
```

## 알려진 이슈

1. **스트리밍 + 참조 이미지**: 동시 사용 불가 (스트리밍 완료 후 파싱 필요)
2. **VLM 이미지 참조**: VLM이 `[REFERENCED_IMAGES: ...]` 태그를 생성하지 않는 경우 빈 리스트 반환

## 향후 작업 제안

1. **웹 UI**: FastAPI + React 기반 챗봇 인터페이스
2. **이미지 렌더링**: 참조 이미지를 응답에 인라인 표시
3. **대화 히스토리**: 멀티턴 대화 지원
4. **평가 시스템**: RAG 응답 품질 평가 메트릭

## 커밋 히스토리

```
095db9a feat: query.py 스트리밍 및 graceful shutdown 지원
38e5e9b feat: query.py 추가 및 프롬프트 튜닝
414bab7 refactor: 스크립트 이름 변경 및 ingest.py 추가
88b0682 feat: EZIS RAG 프로젝트 초기 설정
```

## 연락처

- 프로젝트 담당: 재솔님
- 회사: 위데이터랩 (WeDataLab)
