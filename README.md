# EZIS RAG

EZIS 제품 매뉴얼을 위한 멀티모달 RAG 시스템. LightRAG + RAGAnything 기반으로 PostgreSQL 백엔드를 통합하여 텍스트, 이미지, 테이블, 방정식을 처리한다.

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         demo-front                               │
│                    (Vanilla JS 채팅 UI)                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP /query/stream
┌──────────────────────────▼──────────────────────────────────────┐
│                     scripts/server.py                            │
│                    (FastAPI @ 9621)                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      RAGAnything                                 │
│         (upstream/raganything - 멀티모달 처리 레이어)             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ MinerU      │  │ Modal       │  │ Query Layer             │  │
│  │ Parser      │  │ Processors  │  │ (VLM + 참조 이미지)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                       LightRAG                                   │
│              (지식 그래프 기반 RAG 엔진)                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     PostgreSQL 16                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ pgvector    │  │ Apache AGE  │  │ KV Tables               │  │
│  │ (벡터 검색) │  │ (그래프 DB) │  │ (문서/캐시)             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 기반 라이브러리

### LightRAG (lightrag-hku)

지식 그래프 기반 RAG 프레임워크. 문서에서 엔티티와 관계를 추출하여 그래프를 구축하고, 이를 활용해 컨텍스트 기반 검색을 수행한다.

- 5가지 쿼리 모드: `local`, `global`, `hybrid`, `naive`, `mix`
- 스토리지 추상화: KV, Vector, Graph, DocStatus

### RAGAnything

LightRAG 위에 멀티모달 처리 레이어를 추가한 확장. MinerU 파서로 PDF를 분석하고, 이미지/테이블/방정식을 별도 프로세서로 처리한다.

- 콘텐츠 타입별 프로세서: Image, Table, Equation, Generic
- Vision Language Model 통합

## 이 프로젝트의 확장

### Upstream 수정 (`upstream/raganything/`)

| 파일 | 수정 내용 |
|------|----------|
| `query.py` | VLM 스트리밍 지원, 참조 이미지 자동 반환 (`return_images=True`) |
| `prompt.py` | JSON 포맷팅 지침 추가 (LaTeX 이스케이프 처리) |
| `processor.py` | `multimodal_processed` 플래그를 metadata JSONB에 저장 |

### PostgreSQL 통합

단일 PostgreSQL 인스턴스에서 4가지 스토리지 백엔드를 통합 운영:

| 스토리지 | 구현 | 용도 |
|---------|------|------|
| KV | `PGKVStorage` | 문서, 청크, LLM 캐시 |
| Vector | `PGVectorStorage` | 임베딩 (pgvector) |
| Graph | `PGGraphStorage` | 엔티티/관계 그래프 (Apache AGE) |
| DocStatus | `PGDocStatusStorage` | 문서 처리 상태 |

### 운영 스크립트 (`scripts/`)

PDF 인덱싱부터 쿼리, 서버 운영, 데이터 정리까지 CLI 도구 제공.

### 웹 프론트엔드 (`demo-front/`)

스트리밍 응답과 참조 이미지를 지원하는 채팅 UI.

## 빠른 시작

```bash
# 1. PostgreSQL 시작
cd docker && docker-compose up -d && cd ..

# 2. 환경 변수 설정
cp env.example .env
# .env 파일에서 OPENAI_API_KEY 설정

# 3. 의존성 설치
uv sync

# 4. 문서 인덱싱
uv run python scripts/ingest.py manuals/

# 5. 서버 실행
uv run python scripts/server.py

# 6. 브라우저 접속
open http://localhost:9621
```

## 스크립트 사용법

### ingest.py - 문서 인덱싱

```bash
# 단일 파일
uv run python scripts/ingest.py manuals/EZIS_Manual.pdf

# 폴더 전체
uv run python scripts/ingest.py manuals/

# 강제 재처리
uv run python scripts/ingest.py manuals/ --force

# 특정 패턴 제외
uv run python scripts/ingest.py manuals/ --exclude backup old
```

### cleanup.py - 데이터 정리

```bash
# 현재 상태 확인 (dry-run)
uv run python scripts/cleanup.py --dry-run

# DB 데이터만 삭제 (테이블 구조 유지)
uv run python scripts/cleanup.py

# 테이블 구조까지 삭제
uv run python scripts/cleanup.py --drop-tables

# 파싱 결과물도 삭제
uv run python scripts/cleanup.py --include-artifacts

# 전체 정리 (DB + rag_storage + output)
uv run python scripts/cleanup.py --all
```

### query.py - CLI 쿼리

```bash
# 단일 질의
uv run python scripts/query.py "EZIS 시스템 소개"

# 스트리밍 응답
uv run python scripts/query.py "Oracle 설정 방법" --stream

# 쿼리 모드 지정
uv run python scripts/query.py "시스템 아키텍처" --mode global

# 대화형 모드
uv run python scripts/query.py --interactive
```

**대화형 모드 명령어:**
- `/mode <mode>` - 쿼리 모드 변경
- `/vlm` - VLM 토글
- `/stream` - 스트리밍 토글
- `/clear` - 화면 정리
- `/quit` - 종료

### server.py - API 서버

```bash
# 기본 실행 (포트 9621)
uv run python scripts/server.py

# 포트 지정
uv run python scripts/server.py --port 8000
```

**엔드포인트:**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/query` | 일반 쿼리 |
| POST | `/query/stream` | 스트리밍 쿼리 |
| GET | `/health` | 헬스 체크 |
| GET | `/storage/info` | 저장소 정보 |

## lightrag-server vs scripts/server.py

| | lightrag-server | scripts/server.py |
|---|----------------|-------------------|
| 패키지 | `lightrag-hku[api]` 내장 | 이 프로젝트 |
| 기능 | 쿼리 전용, WebUI 포함 | 멀티모달 지원, 참조 이미지 |
| 문서 삽입 | 텍스트만 | RAGAnything 멀티모달 |
| 용도 | 빠른 테스트, 그래프 시각화 | 프로덕션 API |

**lightrag-server 실행:**

```bash
set -a && source .env && set +a

lightrag-server \
  --host 0.0.0.0 \
  --port 9621 \
  --working-dir ./rag_storage \
  --llm-binding openai \
  --embedding-binding openai
```

## 환경 변수

```bash
# OpenAI API
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
VISION_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=pgvector
POSTGRES_PASSWORD=pgvector
POSTGRES_DATABASE=ezis_rag

# Workspace (RAGAnything과 lightrag-server 공통)
WORKSPACE=default

# 스토리지 백엔드 (RAGAnything용)
KV_STORAGE=PGKVStorage
VECTOR_STORAGE=PGVectorStorage
GRAPH_STORAGE=PGGraphStorage
DOC_STATUS_STORAGE=PGDocStatusStorage

# 스토리지 백엔드 (lightrag-server용)
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
```

## 디렉토리 구조

```
RAG-Anything-1/
├── upstream/               # RAGAnything 1.2.8 (editable, 수정됨)
│   └── raganything/
├── scripts/                # 운영 스크립트
│   ├── ingest.py          # 문서 인덱싱
│   ├── cleanup.py         # 데이터 정리
│   ├── query.py           # CLI 쿼리
│   └── server.py          # FastAPI 서버
├── demo-front/             # 웹 프론트엔드
│   ├── index.html
│   └── app.js
├── docker/                 # PostgreSQL 컨테이너
│   ├── docker-compose.yml
│   ├── Dockerfile.postgres
│   └── init_postgres.sql
├── manuals/                # EZIS 제품 매뉴얼 (PDF)
├── docs/                   # 프로젝트 문서
│   ├── POSTGRES_RAG_GUIDE.md
│   ├── POSTGRES_MIGRATION.md
│   └── HANDOFF.md
├── output/                 # 파싱 결과물
├── rag_storage/            # LightRAG 로컬 캐시
├── .env                    # 환경 변수
├── pyproject.toml
└── requirements.txt
```

## 관련 문서

- [PostgreSQL RAG 가이드](docs/POSTGRES_RAG_GUIDE.md) - DB 설정 및 운영
- [데이터베이스 마이그레이션](docs/POSTGRES_MIGRATION.md) - 백업/복원
- [프로젝트 인수인계](docs/HANDOFF.md) - 상세 구현 내용
