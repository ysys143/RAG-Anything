# EZIS RAG Project Documentation Index

## Overview

EZIS RAG는 위데이터랩의 EZIS 제품 매뉴얼을 위한 멀티모달 RAG 시스템이다. LightRAG와 RAGAnything을 기반으로 PostgreSQL 백엔드를 통합하여 텍스트, 이미지, 테이블, 방정식을 처리한다.

---

## Documentation Structure

```
docs/
├── INDEX.md                 # 이 문서 (진입점)
├── BACKLOGS.md              # 백로그 및 할 일
│
├── guides/                  # 운영 가이드
│   ├── POSTGRES_RAG_GUIDE.md
│   ├── POSTGRES_MIGRATION.md
│   └── PROJECT_SETUP.md
│
├── reference/               # 기술 참조
│   ├── API_REFERENCE.md
│   └── ARCHITECTURE.md
│
├── handoff/                 # 인수인계
│   ├── HANDOFF.md
│   ├── MIGRATION_PLAN.md
│   └── MIGRATION_STATUS_20250108.md
│
└── reports/                 # 리포트
    └── LATENCY_BY_TYPE_260108.md
```

---

## Quick Links

### 시작하기
| 문서 | 설명 |
|------|------|
| [README.md](../README.md) | 프로젝트 개요 및 빠른 시작 |
| [PROJECT_SETUP.md](./guides/PROJECT_SETUP.md) | 개발 환경 설정 |

### 운영 가이드
| 문서 | 설명 |
|------|------|
| [POSTGRES_RAG_GUIDE.md](./guides/POSTGRES_RAG_GUIDE.md) | PostgreSQL 백엔드 운영 |
| [POSTGRES_MIGRATION.md](./guides/POSTGRES_MIGRATION.md) | 데이터베이스 백업/복원 |
| [MIGRATION_PLAN.md](./handoff/MIGRATION_PLAN.md) | 컨테이너화 배포 계획 |

### 기술 참조
| 문서 | 설명 |
|------|------|
| [API_REFERENCE.md](./reference/API_REFERENCE.md) | HTTP API 및 Python API |
| [ARCHITECTURE.md](./reference/ARCHITECTURE.md) | 시스템 아키텍처 상세 |

### 인수인계
| 문서 | 설명 |
|------|------|
| [HANDOFF.md](./handoff/HANDOFF.md) | 프로젝트 인수인계 문서 |
| [MIGRATION_STATUS_20250108.md](./handoff/MIGRATION_STATUS_20250108.md) | 마이그레이션 진행 상황 |

---

## Architecture Overview

```
                          ┌─────────────────────────────┐
                          │      demo-front (UI)        │
                          │    Vanilla JS Chat UI       │
                          └──────────────┬──────────────┘
                                         │ HTTP
                          ┌──────────────▼──────────────┐
                          │    scripts/server.py        │
                          │     FastAPI @ 9621          │
                          └──────────────┬──────────────┘
                                         │
┌────────────────────────────────────────┼────────────────────────────────────────┐
│                               RAGAnything                                        │
│  ┌──────────────┐  ┌─────────────────┐  ┌─────────────────────────────────────┐│
│  │ MinerU/      │  │ Modal           │  │ Query Layer                         ││
│  │ Docling      │  │ Processors      │  │ (VLM Enhanced + Image References)  ││
│  │ Parser       │  │ (Image/Table/   │  │                                     ││
│  │              │  │  Equation)      │  │                                     ││
│  └──────────────┘  └─────────────────┘  └─────────────────────────────────────┘│
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │         LightRAG            │
                          │   Knowledge Graph RAG       │
                          └──────────────┬──────────────┘
                                         │
┌────────────────────────────────────────┼────────────────────────────────────────┐
│                             PostgreSQL 16                                        │
│  ┌──────────────┐  ┌─────────────────┐  ┌─────────────────────────────────────┐│
│  │ pgvector     │  │ Apache AGE      │  │ KV Tables                           ││
│  │ (Vector DB)  │  │ (Graph DB)      │  │ (Documents, Cache)                  ││
│  └──────────────┘  └─────────────────┘  └─────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
RAG-Anything-1/
├── scripts/                    # 운영 스크립트
│   ├── server.py              # FastAPI 서버 (Gemini + OpenAI Embedding)
│   ├── ingest.py              # 문서 인덱싱 (OpenAI)
│   ├── query.py               # CLI 쿼리 (Gemini)
│   ├── cleanup.py             # DB 정리
│   ├── benchmark_query.py     # 쿼리 성능 벤치마크
│   └── test_server_e2e.py     # E2E 테스트
│
├── upstream/                   # RAGAnything 1.2.8 (수정됨)
│   └── raganything/
│       ├── raganything.py     # 메인 클래스 (QueryMixin + ProcessorMixin + BatchMixin)
│       ├── config.py          # 설정 클래스 (환경변수 지원)
│       ├── query.py           # 쿼리 로직 (VLM 스트리밍, 참조 이미지)
│       ├── processor.py       # 문서 처리 (파싱, 캐싱, 청킹)
│       ├── modalprocessors.py # 멀티모달 프로세서 (Image/Table/Equation)
│       ├── parser.py          # MinerU/Docling 파서
│       ├── prompt.py          # 프롬프트 템플릿
│       ├── batch.py           # 배치 처리
│       ├── batch_parser.py    # 배치 파싱
│       └── utils.py           # 유틸리티
│
├── docker/                     # Docker 설정
│   ├── docker-compose.yml     # PostgreSQL 컨테이너 (standalone)
│   ├── Dockerfile.postgres    # pgvector + Apache AGE 이미지
│   └── init_postgres.sql      # DB 초기화 스크립트
│
├── demo-front/                 # 웹 프론트엔드
│   ├── index.html             # 채팅 UI
│   └── app.js                 # 스트리밍 + 이미지 렌더링
│
├── docs/                       # 문서
├── manuals/                    # EZIS 제품 매뉴얼 (PDF)
├── output/                     # 파싱 결과물 (이미지, 마크다운)
├── rag_storage/                # LightRAG 로컬 캐시 (미사용, PG 백엔드)
│
├── docker-compose.yml          # 전체 스택 (앱 + DB)
├── Dockerfile                  # FastAPI 앱 컨테이너
├── .env                        # 환경변수 (gitignore)
├── CLAUDE.md                   # Claude Code 지침
└── README.md                   # 프로젝트 소개
```

---

## Core Components

### 1. RAGAnything Class (`upstream/raganything/raganything.py`)

메인 진입점. Mixin 패턴으로 기능을 조합한다.

```python
@dataclass
class RAGAnything(QueryMixin, ProcessorMixin, BatchMixin):
    lightrag: Optional[LightRAG]          # LightRAG 인스턴스
    llm_model_func: Optional[Callable]    # LLM 함수
    vision_model_func: Optional[Callable] # VLM 함수
    embedding_func: Optional[Callable]    # 임베딩 함수
    config: Optional[RAGAnythingConfig]   # 설정
    lightrag_kwargs: Dict[str, Any]       # LightRAG 파라미터
```

**주요 메서드:**
- `_ensure_lightrag_initialized()`: LightRAG 초기화
- `finalize_storages()`: 스토리지 정리
- `_initialize_processors()`: 멀티모달 프로세서 초기화

### 2. QueryMixin (`upstream/raganything/query.py`)

세 가지 쿼리 모드를 제공한다.

| 메서드 | 설명 | 반환 |
|--------|------|------|
| `aquery()` | 텍스트 쿼리 (VLM 자동 활성화) | `str` / `dict` |
| `aquery_with_multimodal()` | 멀티모달 콘텐츠 포함 쿼리 | `str` |
| `aquery_vlm_enhanced()` | VLM 강화 쿼리 (이미지 참조) | `str` / `dict` |

**VLM Enhanced 플로우:**
1. LightRAG에서 컨텍스트 추출 (`only_need_prompt=True`)
2. 이미지 경로 감지 및 base64 인코딩
3. VLM에 멀티모달 메시지 전송
4. `[REFERENCED_IMAGES: 1, 3]` 태그 파싱

### 3. ProcessorMixin (`upstream/raganything/processor.py`)

문서 파싱 및 처리를 담당한다.

**주요 기능:**
- 파싱 결과 캐싱 (`parse_cache`)
- 콘텐츠 기반 `doc_id` 생성
- 멀티모달 콘텐츠 분리 및 처리
- `multimodal_processed` 플래그 관리

### 4. Modal Processors (`upstream/raganything/modalprocessors.py`)

콘텐츠 타입별 전문 프로세서.

| 프로세서 | 용도 | 모델 |
|----------|------|------|
| `ImageModalProcessor` | 이미지 분석 | Vision Model |
| `TableModalProcessor` | 테이블 해석 | LLM |
| `EquationModalProcessor` | LaTeX 수식 | LLM |
| `GenericModalProcessor` | 기타 콘텐츠 | LLM |

**ContextExtractor:**
주변 콘텐츠를 추출하여 프로세서에 컨텍스트를 제공한다.

---

## Scripts Reference

### server.py (FastAPI 서버)

```bash
uv run python scripts/server.py [--host HOST] [--port PORT]
```

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 헬스 체크 |
| `/storage/info` | GET | 스토리지 백엔드 정보 |
| `/query` | POST | 일반 쿼리 |
| `/query/stream` | POST | 스트리밍 쿼리 |

**LLM 설정:**
- LLM + Vision: Gemini (`gemini-3-flash-preview`)
- Embedding: OpenAI (`text-embedding-3-small`)

### ingest.py (문서 인덱싱)

```bash
uv run python scripts/ingest.py <path> [options]
```

| 옵션 | 설명 |
|------|------|
| `--force` | 이미 처리된 문서도 재처리 |
| `--exclude <patterns>` | 제외 패턴 (기본: backup) |
| `--no-exclude` | 모든 파일 포함 |

**LLM 설정:**
- LLM: OpenAI (`gpt-4.1-mini`)
- Vision: OpenAI (`gpt-4.1`)
- Embedding: OpenAI (`text-embedding-3-small`)

### query.py (CLI 쿼리)

```bash
uv run python scripts/query.py <query> [options]
uv run python scripts/query.py --interactive
```

| 옵션 | 설명 |
|------|------|
| `--mode` | local/global/hybrid/naive/mix |
| `--stream` | 스트리밍 응답 |
| `--no-vlm` | VLM 비활성화 |
| `--interactive` | 대화형 모드 |

**대화형 명령어:**
- `/mode <mode>`: 모드 변경
- `/vlm`: VLM 토글
- `/stream`: 스트리밍 토글
- `/quit`: 종료

### cleanup.py (데이터 정리)

```bash
uv run python scripts/cleanup.py [options]
```

| 옵션 | 삭제 대상 |
|------|----------|
| (기본) | DB 테이블 데이터 |
| `--drop-tables` | 테이블 구조 |
| `--include-artifacts` | output/ 파싱 결과물 |
| `--all` | 전체 (DB + rag_storage + output) |

---

## Configuration

### Environment Variables

```bash
# LLM - Gemini (server.py, query.py)
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-3-flash-preview

# LLM - OpenAI (ingest.py)
OPENAI_API_KEY=your-key
LLM_MODEL=gpt-4.1-mini
VISION_MODEL=gpt-4.1

# Embedding - OpenAI (all scripts)
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=pgvector
POSTGRES_PASSWORD=pgvector
POSTGRES_DATABASE=ezis_rag

# Parser
PARSER=mineru
PARSE_METHOD=auto
WORKING_DIR=./rag_storage
OUTPUT_DIR=./output
```

### RAGAnythingConfig Fields

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `working_dir` | `./rag_storage` | RAG 스토리지 디렉토리 |
| `parser` | `mineru` | 파서 선택 (mineru/docling) |
| `parse_method` | `auto` | 파싱 방식 (auto/ocr/txt) |
| `enable_image_processing` | `True` | 이미지 처리 활성화 |
| `enable_table_processing` | `True` | 테이블 처리 활성화 |
| `enable_equation_processing` | `True` | 수식 처리 활성화 |
| `context_window` | `1` | 컨텍스트 윈도우 크기 |
| `max_context_tokens` | `2000` | 최대 컨텍스트 토큰 |

---

## Database Schema

LightRAG가 자동 생성하는 PostgreSQL 테이블:

| 테이블 | 용도 |
|--------|------|
| `lightrag_doc_full` | 원본 문서 |
| `lightrag_doc_chunks` | 텍스트 청크 |
| `lightrag_doc_status` | 문서 처리 상태 + `multimodal_processed` 플래그 |
| `lightrag_vdb_chunks` | 청크 벡터 임베딩 (pgvector) |
| `lightrag_vdb_entity` | 엔티티 벡터 임베딩 |
| `lightrag_vdb_relation` | 관계 벡터 임베딩 |
| `lightrag_full_entities` | 추출된 엔티티 |
| `lightrag_full_relations` | 엔티티 간 관계 |
| `lightrag_llm_cache` | LLM 응답 캐시 |
| `lightrag_entity_chunks` | 엔티티-청크 매핑 |
| `lightrag_relation_chunks` | 관계-청크 매핑 |

**Apache AGE 그래프:**
- `chunk_entity_relation`: 엔티티/관계 그래프

---

## Query Modes

| Mode | 검색 방식 | 적합한 질문 |
|------|----------|------------|
| `naive` | 단순 벡터 유사도 | 키워드 매칭, 사실 확인 |
| `local` | 엔티티 중심 | "X가 뭐야?", "Y의 정의는?" |
| `global` | 관계/커뮤니티 중심 | "전체 흐름은?", "주요 주제들은?" |
| `hybrid` | local + global | 균형 잡힌 답변 |
| `mix` | hybrid + naive (기본) | 대부분의 질문에 권장 |

---

## Upstream Modifications

원본 RAGAnything에서 수정된 부분:

| 파일 | 수정 내용 |
|------|----------|
| `query.py` | VLM 스트리밍 지원, `return_images=True`로 참조 이미지 반환 |
| `prompt.py` | JSON 포맷팅 지침 추가 (LaTeX 이스케이프) |
| `processor.py` | `multimodal_processed` 플래그를 metadata JSONB에 저장 |

---

## Related Documents

- **운영 가이드**: [guides/POSTGRES_RAG_GUIDE.md](./guides/POSTGRES_RAG_GUIDE.md)
- **마이그레이션**: [guides/POSTGRES_MIGRATION.md](./guides/POSTGRES_MIGRATION.md)
- **배포 계획**: [MIGRATION_PLAN.md](./handoff/MIGRATION_PLAN.md)
- **백로그**: [BACKLOGS.md](./BACKLOGS.md)

---

## Version History

| 날짜 | 버전 | 주요 변경 |
|------|------|----------|
| 2025-01-09 | 1.3 | Gemini LLM 마이그레이션, 전체 스택 컨테이너화 |
| 2025-01-08 | 1.2 | 마이그레이션 계획 수립 |
| 2025-12-31 | 1.1 | PostgreSQL 백엔드 통합 |
| 2025-12-30 | 1.0 | VLM 스트리밍, 참조 이미지 반환, Graceful Shutdown |

---

*Last Updated: 2026-01-13*
