# EZIS RAG Project Setup

## 프로젝트 개요

EZIS 제품 매뉴얼을 위한 멀티모달 RAG 시스템. RAG-Anything 기반으로 PostgreSQL 백엔드를 사용하여 KV, Graph, Vector 저장소를 통합 관리한다.

## 프로젝트 구조

```
ezis-rag/
├── .venv/                    # Python 3.12.11 가상환경
├── docker/                   # PostgreSQL 컨테이너 설정
│   ├── docker-compose.yml
│   ├── Dockerfile.postgres
│   └── init_postgres.sql
├── manuals/                  # EZIS 제품 매뉴얼 (PDF)
│   ├── EZIS_Oracle_Manual_*.pdf
│   ├── EZIS_PostgreSQL_Manual_*.pdf
│   ├── EZIS_SQLServer_Manual_*.pdf
│   ├── EZIS_Maria_MySQL_Manual_*.pdf
│   ├── EZIS_Dashboard_*.pdf
│   ├── EZIS_Audit_Manual_*.pdf
│   └── backup/               # 이전 버전 매뉴얼
├── docs/                     # 프로젝트 문서
├── requirements.txt          # Python 의존성
├── .gitignore
├── CLAUDE.md                 # Claude Code 가이드
│
└── upstream/                 # RAG-Anything 소스 (editable install)
    ├── raganything/          # 핵심 라이브러리
    ├── examples/             # 예제 코드
    ├── docs/                 # 원본 문서
    └── pyproject.toml
```

## 설치 완료 항목

### Python 환경

- Python 3.12.11
- 가상환경: `.venv/`
- 패키지 관리: `uv`

### 설치된 주요 패키지

| 패키지 | 버전 | 설치 방식 |
|-------|------|----------|
| raganything | 1.2.8 | editable (`-e ./upstream[all]`) |
| lightrag-hku | 1.4.9.8 | PyPI |
| psycopg2-binary | 2.9.11 | PyPI |
| asyncpg | 0.31.0 | PyPI |
| mineru | 2.6.8 | 의존성으로 설치 |

### PostgreSQL 컨테이너

| 항목 | 값 |
|-----|---|
| Container Name | postgres-rag |
| Image | docker-postgres-rag (커스텀 빌드) |
| Port | 5432 |
| Database | ezis_rag |
| User | pgvector |
| Password | pgvector |

### PostgreSQL 확장

| 확장 | 버전 | 용도 |
|-----|------|-----|
| pgvector | 0.8.1 | 벡터 유사도 검색 (VDB) |
| Apache AGE | 1.5.0 | 그래프 데이터베이스 |

## 환경 변수 설정

`.env` 파일 생성 필요:

```bash
# LLM API
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1  # optional

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=pgvector
POSTGRES_PASSWORD=pgvector
POSTGRES_DATABASE=ezis_rag

# Parser
PARSER=mineru
PARSE_METHOD=auto
```

## 사용 방법

### Docker 컨테이너 시작

```bash
cd docker
docker-compose up -d
```

### 확장 상태 확인

```bash
docker exec postgres-rag psql -U pgvector -d ezis_rag -c "\dx"
```

### Python 환경 활성화

```bash
source .venv/bin/activate
```

### RAG-Anything 사용 예시

```python
import os
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# 환경 변수 설정
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_PORT"] = "5432"
os.environ["POSTGRES_USER"] = "pgvector"
os.environ["POSTGRES_PASSWORD"] = "pgvector"
os.environ["POSTGRES_DATABASE"] = "ezis_rag"

async def main():
    api_key = os.getenv("OPENAI_API_KEY")

    # LLM 함수 정의
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            **kwargs,
        )

    # 임베딩 함수 정의
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
        ),
    )

    # RAGAnything 초기화 (PostgreSQL 백엔드)
    rag = RAGAnything(
        config=RAGAnythingConfig(
            working_dir="./rag_storage",
            parser="mineru",
        ),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "kv_storage": "PGKVStorage",
            "vector_storage": "PGVectorStorage",
            "graph_storage": "PGGraphStorage",
        }
    )

    # 문서 처리
    await rag.process_document_complete("manuals/EZIS_Oracle_Manual_20251216.pdf")

    # 쿼리
    result = await rag.aquery("EZIS Oracle 모니터링 방법은?")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Git 상태

- 저장소: 새로 초기화됨
- 브랜치: main
- Remote: 미설정

## 다음 단계

1. [ ] `.env` 파일 생성 및 API 키 설정
2. [ ] 매뉴얼 문서 인덱싱 테스트
3. [ ] 쿼리 성능 검증
4. [ ] Git remote 설정 및 초기 커밋
