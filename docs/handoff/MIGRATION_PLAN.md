# RAG-Anything 마이그레이션 및 컨테이너화 배포 계획

## 개요

현재 로컬에서 운영 중인 RAG-Anything 프로젝트를 `ezis_manual_chatbot` 저장소로 이전하고, 전체 스택을 컨테이너화하여 192.168.0.47 서버에 배포한다.

### 마이그레이션 목표

| 항목 | 현재 | 목표 |
|------|------|------|
| 저장소 | RAG-Anything-1 (로컬) | WeDataLab-AI-Lab/ezis_manual_chatbot |
| 배포 환경 | 로컬 Docker | 192.168.0.47 서버 |
| 컨테이너화 | DB만 컨테이너 | 전체 스택 (앱 + DB) |
| Git History | 보존 | 보존 (remote 변경 방식) |

### 현재 인프라 현황

**로컬 환경:**
- PostgreSQL 컨테이너: `postgres-rag` (pgvector + Apache AGE)
- 데이터베이스: `ezis_rag` (721 MB, 11개 LightRAG 테이블)
- 볼륨: `docker_postgres_data`
- 포트: 5432

**0.47 서버:**
- Docker + docker-compose 설치됨
- 사용 중인 포트: 48000, 43306, 1511, 5512, 910
- 기존 `~/rag_anything` 폴더 존재 (무시하고 덮어쓰기)

---

## Phase 1: 로컬 PostgreSQL 데이터 백업

### 1.1 SQL 덤프 생성

```bash
# pg_dump로 스키마 + 데이터 백업 (custom format)
docker exec postgres-rag pg_dump -U pgvector -d ezis_rag \
  --format=custom \
  --verbose \
  --file=/tmp/ezis_rag_backup.dump

# 덤프 파일을 로컬로 복사
docker cp postgres-rag:/tmp/ezis_rag_backup.dump ./backup/ezis_rag_backup.dump
```

### 1.2 볼륨 백업 (추가 안전장치)

```bash
# 볼륨 전체를 tar로 백업
docker run --rm \
  -v docker_postgres_data:/data:ro \
  -v $(pwd)/backup:/backup \
  alpine tar cvf /backup/postgres_volume_backup.tar -C /data .
```

### 1.3 백업 검증

```bash
# 덤프 파일 크기 확인
ls -lh ./backup/ezis_rag_backup.dump

# 체크섬 생성 (전송 후 검증용)
shasum -a 256 ./backup/ezis_rag_backup.dump > ./backup/ezis_rag_backup.dump.sha256
```

### 백업 방식 선택 이유

| 방식 | 장점 | 단점 |
|------|------|------|
| pg_dump (custom) | 압축됨, 선택적 복원 가능, 병렬 복원 지원 | 복원 시 pg_restore 필요 |
| 볼륨 tar | 완전한 물리적 백업, 빠른 복원 | PostgreSQL 버전 의존성 |

**권장**: 두 방식 모두 수행하여 이중 안전장치 확보

---

## Phase 2: 커스텀 Docker 이미지 전송

### 2.1 이미지 Export

```bash
# 커스텀 PostgreSQL 이미지 저장
docker save docker-postgres-rag -o ./backup/postgres-rag-image.tar

# 이미지 크기 확인
ls -lh ./backup/postgres-rag-image.tar
```

### 2.2 0.47 서버로 전송

```bash
# SCP로 파일 전송
scp -P 910 ./backup/ezis_rag_backup.dump sh-tech@192.168.0.47:~/rag_anything/backup/
scp -P 910 ./backup/postgres-rag-image.tar sh-tech@192.168.0.47:~/rag_anything/

# 체크섬 검증
ssh sh-tech@192.168.0.47 -p 910 "shasum -a 256 ~/rag_anything/backup/ezis_rag_backup.dump"
```

### 2.3 0.47에서 이미지 로드

```bash
ssh sh-tech@192.168.0.47 -p 910 "docker load -i ~/rag_anything/postgres-rag-image.tar"
```

### 커스텀 이미지가 필요한 이유

기본 `pgvector/pgvector` 이미지에는 **Apache AGE** 확장이 없다. LightRAG의 `PGGraphStorage`가 Apache AGE를 사용하므로 커스텀 이미지가 필수이다.

```dockerfile
# docker/Dockerfile.postgres 요약
FROM pgvector/pgvector:pg16
# Apache AGE 1.5.0 빌드 및 설치
RUN git clone --branch release/PG16/1.5.0 https://github.com/apache/age.git ...
```

---

## Phase 3: Git History 보존 저장소 이전

### 3.1 새 Remote 추가

```bash
cd /Users/jaesolshin/Documents/GitHub/RAG-Anything-1

# 새 저장소를 remote로 추가
git remote add ezis https://github.com/WeDataLab-AI-Lab/ezis_manual_chatbot.git

# remote 확인
git remote -v
```

### 3.2 모든 브랜치/태그 Push

```bash
# 모든 브랜치 push
git push ezis --all

# 모든 태그 push
git push ezis --tags

# (필요시) 기존 보일러플레이트 덮어쓰기
git push ezis --all --force
```

### 3.3 로컬 저장소 정리 (선택)

```bash
# 기존 origin을 backup으로 변경
git remote rename origin backup

# ezis를 origin으로 변경
git remote rename ezis origin
```

---

## Phase 4: 애플리케이션 컨테이너화

### 4.1 디렉토리 구조

```
ezis_manual_chatbot/
├── docker-compose.yml          # 전체 스택 정의
├── docker-compose.override.yml # 로컬 개발용 오버라이드 (선택)
├── Dockerfile                  # FastAPI 앱 컨테이너
├── docker/
│   ├── Dockerfile.postgres     # PostgreSQL + pgvector + AGE
│   └── init_postgres.sql       # DB 초기화 스크립트
├── scripts/
│   └── server.py               # FastAPI 서버 엔트리포인트
├── raganything/                # 앱 코드
├── pyproject.toml
├── .env.example
└── .env                        # (gitignore) 실제 환경변수
```

### 4.2 FastAPI Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY pyproject.toml .
RUN pip install --no-cache-dir uv && \
    uv pip install --system -e .

# 앱 코드 복사
COPY raganything/ ./raganything/
COPY scripts/ ./scripts/

# 환경변수 기본값
ENV PYTHONUNBUFFERED=1
ENV WORKING_DIR=/app/rag_storage
ENV OUTPUT_DIR=/app/output

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 실행
CMD ["python", "scripts/server.py", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.3 docker-compose.yml (전체 스택)

```yaml
services:
  # FastAPI 애플리케이션
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ezis-rag-app
    restart: unless-stopped
    ports:
      - "38000:8000"              # 외부 38000 -> 내부 8000
    depends_on:
      postgres-rag:
        condition: service_healthy
    environment:
      # PostgreSQL
      POSTGRES_HOST: postgres-rag
      POSTGRES_PORT: 5432
      POSTGRES_USER: pgvector
      POSTGRES_PASSWORD: pgvector
      POSTGRES_DATABASE: ezis_rag
      # LLM (Gemini)
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      GEMINI_MODEL: ${GEMINI_MODEL:-gemini-3-flash-preview}
      # Embedding (OpenAI)
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      EMBEDDING_MODEL: ${EMBEDDING_MODEL:-text-embedding-3-small}
      EMBEDDING_DIM: ${EMBEDDING_DIM:-1536}
      # Parser
      PARSER: ${PARSER:-mineru}
      PARSE_METHOD: ${PARSE_METHOD:-auto}
    volumes:
      - rag_storage:/app/rag_storage
      - output_data:/app/output
    networks:
      - ezis-network

  # PostgreSQL + pgvector + Apache AGE
  postgres-rag:
    build:
      context: ./docker
      dockerfile: Dockerfile.postgres
    image: docker-postgres-rag      # 빌드 후 이미지 이름
    container_name: ezis-postgres-rag
    restart: unless-stopped
    ports:
      - "35432:5432"                # 외부 35432 -> 내부 5432
    environment:
      POSTGRES_USER: pgvector
      POSTGRES_PASSWORD: pgvector
      POSTGRES_DB: ezis_rag
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/init_postgres.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U pgvector -d ezis_rag"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - ezis-network

volumes:
  postgres_data:
    name: ezis_postgres_data
  rag_storage:
    name: ezis_rag_storage
  output_data:
    name: ezis_output_data

networks:
  ezis-network:
    name: ezis-network
    driver: bridge
```

### 4.4 환경변수 템플릿 (.env.example)

```bash
# LLM Configuration (Gemini)
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash

# Embedding Configuration (OpenAI)
OPENAI_API_KEY=your-openai-api-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

# Parser Configuration
PARSER=mineru
PARSE_METHOD=auto

# PostgreSQL (컨테이너 내부 통신용, 수정 불필요)
# POSTGRES_HOST=postgres-rag
# POSTGRES_PORT=5432
# POSTGRES_USER=pgvector
# POSTGRES_PASSWORD=pgvector
# POSTGRES_DATABASE=ezis_rag
```

---

## Phase 5: 0.47 서버 배포 및 데이터 복원

### 5.1 저장소 클론

```bash
ssh sh-tech@192.168.0.47 -p 910

cd ~
rm -rf rag_anything  # 기존 폴더 제거
git clone https://github.com/WeDataLab-AI-Lab/ezis_manual_chatbot.git rag_anything
cd rag_anything
```

### 5.2 환경변수 설정

```bash
cp .env.example .env
# .env 파일 편집하여 실제 API 키 입력
nano .env
```

### 5.3 컨테이너 시작 (DB만 먼저)

```bash
# PostgreSQL 컨테이너만 먼저 시작
docker-compose up -d postgres-rag

# 컨테이너 상태 확인
docker-compose ps
docker logs ezis-postgres-rag
```

### 5.4 데이터 복원

```bash
# 백업 파일을 컨테이너로 복사
docker cp ~/rag_anything/backup/ezis_rag_backup.dump ezis-postgres-rag:/tmp/

# pg_restore로 데이터 복원
docker exec ezis-postgres-rag pg_restore \
  -U pgvector \
  -d ezis_rag \
  --verbose \
  --clean \
  --if-exists \
  /tmp/ezis_rag_backup.dump

# 복원 후 임시 파일 정리
docker exec ezis-postgres-rag rm /tmp/ezis_rag_backup.dump
```

### 5.5 앱 컨테이너 시작

```bash
# 전체 스택 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f app
```

---

## Phase 6: 검증 및 테스트

### 6.1 DB 검증

```bash
# 테이블 존재 확인
docker exec ezis-postgres-rag psql -U pgvector -d ezis_rag -c "\dt"

# 레코드 수 비교 (로컬과 동일해야 함)
docker exec ezis-postgres-rag psql -U pgvector -d ezis_rag -c "
SELECT
  'lightrag_doc_full' as table_name, count(*) as count FROM lightrag_doc_full
UNION ALL
SELECT 'lightrag_doc_chunks', count(*) FROM lightrag_doc_chunks
UNION ALL
SELECT 'lightrag_vdb_chunks', count(*) FROM lightrag_vdb_chunks
UNION ALL
SELECT 'lightrag_full_entities', count(*) FROM lightrag_full_entities
UNION ALL
SELECT 'lightrag_full_relations', count(*) FROM lightrag_full_relations;
"

# 벡터 데이터 존재 확인
docker exec ezis-postgres-rag psql -U pgvector -d ezis_rag -c "
SELECT count(*) as vectors_count
FROM lightrag_vdb_chunks
WHERE embedding IS NOT NULL;
"

# Apache AGE 그래프 확인
docker exec ezis-postgres-rag psql -U pgvector -d ezis_rag -c "
LOAD 'age';
SET search_path = ag_catalog, public;
SELECT * FROM ag_graph;
"
```

### 6.2 API 검증

```bash
# 헬스체크
curl http://192.168.0.47:38000/health

# 저장소 정보 확인
curl http://192.168.0.47:38000/storage/info

# 테스트 쿼리
curl -X POST http://192.168.0.47:38000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "테스트 질문", "mode": "hybrid"}'
```

### 6.3 검증 체크리스트

| 항목 | 확인 방법 | 예상 결과 |
|------|----------|----------|
| DB 컨테이너 실행 | `docker ps` | ezis-postgres-rag Running |
| App 컨테이너 실행 | `docker ps` | ezis-rag-app Running |
| 테이블 개수 | `\dt` | 11개 테이블 |
| 레코드 수 | SQL COUNT | 로컬과 동일 |
| 벡터 데이터 | embedding NOT NULL | 0보다 큼 |
| API 헬스체크 | /health | {"status": "ok"} |
| 쿼리 테스트 | /query | 정상 응답 |

---

## 포트 매핑 정리

| 서비스 | 컨테이너 내부 포트 | 외부 노출 포트 | 비고 |
|--------|-------------------|---------------|------|
| FastAPI App | 8000 | **38000** | API 서버 |
| PostgreSQL | 5432 | **35432** | DB 직접 접근용 |

### 접근 URL

- API: `http://192.168.0.47:38000`
- DB (외부): `postgresql://pgvector:pgvector@192.168.0.47:35432/ezis_rag`
- DB (내부): `postgresql://pgvector:pgvector@postgres-rag:5432/ezis_rag`

---

## 롤백 계획

### 데이터 복원 실패 시

```bash
# 볼륨 백업에서 복원
docker-compose down
docker volume rm ezis_postgres_data

# 볼륨 재생성 및 tar에서 복원
docker volume create ezis_postgres_data
docker run --rm \
  -v ezis_postgres_data:/data \
  -v ~/rag_anything/backup:/backup \
  alpine tar xvf /backup/postgres_volume_backup.tar -C /data

docker-compose up -d
```

### 전체 롤백

```bash
# 0.47에서 컨테이너 중지
docker-compose down -v

# 로컬에서 기존 환경 유지
# (로컬 postgres-rag 컨테이너는 그대로 유지됨)
```

---

## 보안 고려사항

1. **API 키 관리**: `.env` 파일은 절대 Git에 커밋하지 않음 (`.gitignore` 필수)
2. **DB 비밀번호**: 프로덕션에서는 `pgvector` 대신 강력한 비밀번호 사용 권장
3. **네트워크**: 필요시 방화벽에서 38000, 35432 포트 제한
4. **CORS**: 프로덕션에서는 `allow_origins=["*"]` 대신 특정 도메인만 허용

---

## 제외 항목

| 항목 | 이유 |
|------|------|
| Alembic | LightRAG가 자체적으로 스키마 관리 |
| 0.47 기존 docker-compose.yml | 완전히 덮어쓰기 |
| CI/CD | 초기 배포 후 별도 설정 예정 |

---

## 예상 소요 시간

| Phase | 작업 | 예상 시간 |
|-------|------|----------|
| 1 | DB 백업 | 5-10분 |
| 2 | 이미지 전송 | 10-15분 |
| 3 | Git 저장소 이전 | 5분 |
| 4 | Dockerfile 작성 | 10분 |
| 5 | 배포 및 복원 | 10분 |
| 6 | 검증 | 5분 |
| **총계** | | **45-55분** |

---

## 문서 정보

- **작성일**: 2025-01-08
- **작성자**: Claude Code (brainstorm session)
- **버전**: 1.0
