# PostgreSQL 기반 RAG-Anything 가이드

## 개요

RAG-Anything + LightRAG를 PostgreSQL 백엔드로 운영하는 방법을 설명한다. pgvector(벡터 검색)와 Apache AGE(그래프 DB)를 활용하여 단일 PostgreSQL 인스턴스에서 KV, Vector, Graph 스토리지를 통합 관리한다.

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG-Anything                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ MinerU      │  │ OpenAI      │  │ Vision Model        │ │
│  │ Parser      │  │ LLM/Embed   │  │ (gpt-4.1)            │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         └────────────────┼─────────────────────┘            │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    LightRAG                           │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐│  │
│  │  │ KV Storage │ │ VDB Storage│ │ Graph Storage      ││  │
│  │  │ PGKVStorage│ │PGVectorStorage│ PGGraphStorage    ││  │
│  │  └─────┬──────┘ └─────┬──────┘ └─────────┬──────────┘│  │
│  └────────┼──────────────┼──────────────────┼───────────┘  │
│           │              │                  │               │
└───────────┼──────────────┼──────────────────┼───────────────┘
            │              │                  │
            ▼              ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     PostgreSQL 16                           │
│  ┌─────────────────┐ ┌─────────────┐ ┌───────────────────┐ │
│  │ KV Tables       │ │ pgvector    │ │ Apache AGE        │ │
│  │ - doc_full      │ │ - vdb_chunks│ │ - chunk_entity_   │ │
│  │ - doc_chunks    │ │ - vdb_entity│ │   relation graph  │ │
│  │ - llm_cache     │ │ - vdb_relation│                   │ │
│  │ - doc_status    │ │             │ │                   │ │
│  └─────────────────┘ └─────────────┘ └───────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Docker 설정

### 컨테이너 구성

`docker/docker-compose.yml`:
```yaml
services:
  postgres-rag:
    build:
      context: .
      dockerfile: Dockerfile.postgres
    container_name: postgres-rag
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: pgvector
      POSTGRES_PASSWORD: pgvector
      POSTGRES_DB: ezis_rag
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init_postgres.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U pgvector -d ezis_rag"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

### 컨테이너 실행

```bash
cd docker
docker-compose up -d

# 상태 확인
docker-compose ps

# 확장 확인
docker exec postgres-rag psql -U pgvector -d ezis_rag -c "\dx"
```

### 설치되는 확장

| 확장 | 버전 | 용도 |
|-----|------|-----|
| pgvector | 0.8.1 | 벡터 유사도 검색 |
| Apache AGE | 1.5.0 | 그래프 데이터베이스 |

## 환경 변수 설정

`.env` 파일:
```bash
# OpenAI API
OPENAI_API_KEY=your_api_key

# PostgreSQL 연결
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=pgvector
POSTGRES_PASSWORD=pgvector
POSTGRES_DATABASE=ezis_rag

# LightRAG 스토리지 백엔드
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
```

## 스크립트 사용법

### 문서 처리 및 쿼리 테스트

```bash
python scripts/raganything_openai_postgres_demo.py
```

이 스크립트는:
1. PostgreSQL 연결 및 테이블 자동 생성
2. PDF 문서 파싱 (MinerU)
3. 텍스트/이미지/테이블 멀티모달 처리
4. 엔티티 및 관계 추출
5. 5가지 쿼리 모드 테스트

### 데이터 정리

```bash
# 현재 상태 확인 (dry-run)
python scripts/cleanup.py --dry-run

# DB 데이터만 삭제 (테이블 구조 유지)
python scripts/cleanup.py

# 테이블 완전 삭제 (임베딩 모델 변경 시)
python scripts/cleanup.py --drop-tables

# output/ 파싱 아티팩트도 삭제
python scripts/cleanup.py --include-artifacts

# 전체 정리 (DB + rag_storage + output)
python scripts/cleanup.py --all
```

| 옵션 | 삭제 대상 |
|-----|----------|
| (기본) | DB 테이블 데이터 (TRUNCATE) |
| `--drop-tables` | DB 테이블 구조까지 삭제 |
| `--include-local` | rag_storage/ 디렉토리 |
| `--include-artifacts` | output/ 파싱 결과물 |
| `--all` | 위 모두 포함 |

## 쿼리 모드

| Mode | 검색 방식 | 적합한 질문 |
|------|----------|------------|
| naive | 단순 벡터 유사도 | 키워드 매칭, 사실 확인 |
| local | 엔티티 중심 | "X가 뭐야?", "Y의 정의는?" |
| global | 관계/커뮤니티 중심 | "전체 흐름은?", "주요 주제들은?" |
| hybrid | local + global | 균형 잡힌 답변 |
| mix | hybrid + naive (기본값) | 대부분의 질문에 권장 |

## 테이블 구조

LightRAG가 자동 생성하는 테이블:

| 테이블 | 용도 |
|-------|-----|
| lightrag_doc_full | 원본 문서 저장 |
| lightrag_doc_chunks | 텍스트 청크 |
| lightrag_doc_status | 문서 처리 상태 |
| lightrag_vdb_chunks | 청크 벡터 임베딩 |
| lightrag_vdb_entity | 엔티티 벡터 임베딩 |
| lightrag_vdb_relation | 관계 벡터 임베딩 |
| lightrag_full_entities | 엔티티 전체 정보 |
| lightrag_full_relations | 관계 전체 정보 |
| lightrag_entity_chunks | 엔티티-청크 매핑 |
| lightrag_relation_chunks | 관계-청크 매핑 |
| lightrag_llm_cache | LLM 응답 캐시 |

## 알려진 제한사항

### HNSW 인덱스 차원 제한

pgvector의 HNSW 인덱스는 2000 차원까지만 지원한다. `text-embedding-3-large`(3072 dim) 사용 시 인덱스 생성이 실패하지만 동작에는 영향 없다.

**대안:**
1. `text-embedding-3-small` (1536 dim) 사용
2. 인덱스 없이 사용 (소규모 데이터에 적합)
3. IVFFlat 인덱스 사용 (차원 제한 없음)

### 로컬 PostgreSQL 충돌

로컬에 PostgreSQL이 설치되어 있으면 포트 충돌이 발생할 수 있다. 확인:

```bash
lsof -i :5432
```

해결:
- 로컬 PostgreSQL 중지: `brew services stop postgresql`
- 또는 Docker 포트 변경: `5433:5432`

## 테스트 결과 예시

```
문서: manuals/test.pdf (626KB)

파싱 결과:
- 총 블록: 36
- 텍스트: 26
- 이미지: 1
- 테이블: 1

추출 결과:
- 엔티티: 51
- 관계: 113

쿼리 테스트: 5개 모드 모두 성공
VLM 이미지 처리: 2개 이미지 처리 완료
```

## lightrag-server 실행

PostgreSQL 백엔드로 LightRAG API 서버 실행:

```bash
# 환경 변수 로드
set -a && source .env && set +a

# 서버 실행
lightrag-server \
  --host 0.0.0.0 \
  --port 9621 \
  --working-dir ./rag_storage \
  --llm-binding openai \
  --embedding-binding openai
```

**참고**: lightrag-server는 문서 쿼리만 지원. 멀티모달 문서 삽입은 Python 스크립트(RAG-Anything) 필요.

## 관련 문서

- [POSTGRES_MIGRATION.md](./POSTGRES_MIGRATION.md) - 데이터베이스 마이그레이션 가이드
