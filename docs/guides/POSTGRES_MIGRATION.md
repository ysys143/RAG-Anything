# PostgreSQL 데이터베이스 마이그레이션 가이드

로컬 PostgreSQL RAG 데이터를 원격 서버로 이전하는 방법.

## 백업 방법 비교

| 구분 | Custom Format (`-F c`) | SQL 덤프 |
|------|------------------------|----------|
| 파일 형식 | 바이너리 압축 | 평문 SQL |
| 파일 크기 | 작음 | 큼 |
| 복원 도구 | `pg_restore` | `psql` |
| 선택적 복원 | 가능 | 수동 편집 필요 |
| 병렬 복원 | `-j N` 옵션 지원 | 불가 |
| 권장 | 대용량, 운영 환경 | 소규모, SQL 확인 필요 시 |

## 스키마 및 AGE 그래프 백업 여부

| 구분 | pg_dump 포함 | 비고 |
|------|-------------|------|
| 테이블 스키마 | O | 자동 포함 |
| 인덱스 (HNSW) | O | 벡터 인덱스 포함 |
| pgvector 데이터 | O | 벡터 컬럼 정상 덤프 |
| AGE 그래프 스키마 | 부분적 | `ag_catalog` 별도 처리 권장 |
| AGE 노드/엣지 | 부분적 | 그래프별 스키마로 저장됨 |

## 전체 백업 (권장)

```bash
# 모든 스키마 포함 (가장 안전)
pg_dump -h localhost -p 5432 -U pgvector -d ezis_rag \
  --no-owner --no-privileges \
  -F c -f ezis_rag_backup.dump
```

특정 스키마만 백업:

```bash
pg_dump -h localhost -p 5432 -U pgvector -d ezis_rag \
  --no-owner --no-privileges \
  --schema=public \
  --schema=ag_catalog \
  --schema=chunk_entity_relation \
  -F c -f ezis_rag_backup.dump
```

## 원격 서버 준비

```bash
# 1. 데이터베이스 생성
psql -h remote-host -U postgres -c "CREATE DATABASE ezis_rag OWNER pgvector;"

# 2. 필수 익스텐션 설치
psql -h remote-host -U postgres -d ezis_rag << 'EOF'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
EOF
```

## 복원

```bash
pg_restore -h remote-host -p 5432 -U pgvector -d ezis_rag \
  --no-owner --no-privileges \
  -c ezis_rag_backup.dump
```

## 연결 설정 변경

`.env` 수정:

```bash
POSTGRES_HOST=remote-host
POSTGRES_PORT=5432
POSTGRES_USER=pgvector
POSTGRES_PASSWORD=secure-password
POSTGRES_DATABASE=ezis_rag
```

## 마이그레이션 체크리스트

| 항목 | 확인 사항 |
|------|----------|
| pgvector 버전 | 로컬/원격 동일 권장 (0.8.x) |
| AGE 버전 | Apache AGE 1.5.0 호환성 확인 |
| 벡터 인덱스 | 복원 후 HNSW 인덱스 상태 확인 |
| 네트워크 | 방화벽, `pg_hba.conf` 원격 접속 허용 |
| SSL | 운영 환경에서는 SSL 연결 권장 |

## 인덱스 재생성 (필요 시)

```sql
-- HNSW 인덱스 재생성
DROP INDEX IF EXISTS idx_lightrag_vdb_chunks_hnsw_cosine;
CREATE INDEX idx_lightrag_vdb_chunks_hnsw_cosine
  ON lightrag_vdb_chunks
  USING hnsw (embedding vector_cosine_ops);

DROP INDEX IF EXISTS idx_lightrag_vdb_entity_hnsw_cosine;
CREATE INDEX idx_lightrag_vdb_entity_hnsw_cosine
  ON lightrag_vdb_entity
  USING hnsw (embedding vector_cosine_ops);

DROP INDEX IF EXISTS idx_lightrag_vdb_relation_hnsw_cosine;
CREATE INDEX idx_lightrag_vdb_relation_hnsw_cosine
  ON lightrag_vdb_relation
  USING hnsw (embedding vector_cosine_ops);
```

## 마이그레이션 검증

```bash
# 테이블 행 수 비교
psql -h localhost -U pgvector -d ezis_rag -c "
  SELECT 'local' as source,
         (SELECT COUNT(*) FROM lightrag_doc_status) as docs,
         (SELECT COUNT(*) FROM lightrag_vdb_chunks) as chunks,
         (SELECT COUNT(*) FROM lightrag_vdb_entity) as entities;
"

psql -h remote-host -U pgvector -d ezis_rag -c "
  SELECT 'remote' as source,
         (SELECT COUNT(*) FROM lightrag_doc_status) as docs,
         (SELECT COUNT(*) FROM lightrag_vdb_chunks) as chunks,
         (SELECT COUNT(*) FROM lightrag_vdb_entity) as entities;
"
```

## 관련 문서

- [POSTGRES_RAG_GUIDE.md](./POSTGRES_RAG_GUIDE.md) - PostgreSQL 기반 RAG 운영 가이드
