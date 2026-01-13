# 마이그레이션 핸드오프 문서 (2025-01-08)

## 완료된 작업

### Phase 1-5: 완료

| Phase | 작업 | 상태 |
|-------|------|------|
| 1 | DB 백업 (pg_dump 271MB + 볼륨 1GB) | 완료 |
| 2 | Docker 이미지 전송 (postgres-rag) | 완료 |
| 3 | Git 저장소 이전 (ezis_manual_chatbot) | 완료 |
| 4 | Dockerfile/docker-compose.yml 작성 | 완료 |
| 5 | DB 복원 (11개 테이블, 전체 데이터) | 완료 |

### DB 복원 확인 완료

| 테이블 | 레코드 수 |
|--------|-----------|
| lightrag_vdb_relation | 25,844 |
| lightrag_vdb_entity | 8,050 |
| lightrag_llm_cache | 5,985 |
| lightrag_vdb_chunks | 2,605 |
| lightrag_doc_chunks | 2,605 |
| lightrag_full_entities | 11 |
| lightrag_full_relations | 11 |

---

## 미완료 작업

### Phase 6: 앱 컨테이너 시작

**문제**: `mineru` CLI가 컨테이너에서 인식되지 않음

**원인**: Dockerfile에서 mineru[core] 설치 후 CLI 바이너리가 production stage로 복사되지 않음

**해결된 Dockerfile** (이미 수정됨):
```dockerfile
# builder stage에서 mineru 명시적 설치
RUN uv pip install --system --no-cache -r requirements.txt && \
    uv pip install --system --no-cache "mineru[core]" && \
    which mineru && mineru --version

# production stage에 OpenCV 런타임 의존성 추가
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
```

---

## 내일 수행할 작업

### 옵션 A: 로컬에서 x86_64 이미지 빌드 후 전송 (권장)

로컬(arm64)에서 x86_64용 이미지를 빌드하여 전송:

```bash
# 1. x86_64용 이미지 빌드 (buildx 사용)
docker buildx build --platform linux/amd64 -t ezis-rag-app:latest --load .

# 2. 이미지 저장
docker save ezis-rag-app:latest -o /tmp/ezis-rag-app.tar

# 3. 0.47로 전송
scp -P 910 /tmp/ezis-rag-app.tar sh-tech@192.168.0.47:~/rag_anything/

# 4. 0.47에서 이미지 로드 및 시작
ssh sh-tech@192.168.0.47 -p 910
cd ~/rag_anything
docker load -i ezis-rag-app.tar
docker-compose up -d app
```

### 옵션 B: 서버에서 직접 빌드 (느리지만 확실)

```bash
ssh sh-tech@192.168.0.47 -p 910
cd ~/rag_anything
docker-compose build app
docker-compose up -d app
```

### 2. 헬스체크

```bash
# API 헬스체크
curl http://localhost:38000/health

# 저장소 정보 확인
curl http://localhost:38000/storage/info

# 테스트 쿼리
curl -X POST http://localhost:38000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "테스트", "mode": "hybrid"}'
```

---

## 서버 정보

| 항목 | 값 |
|------|-----|
| 서버 | 192.168.0.47 |
| SSH 포트 | 910 |
| 사용자 | sh-tech |
| 프로젝트 경로 | ~/rag_anything |
| API 포트 | 38000 |
| DB 포트 | 35432 |

### 현재 컨테이너 상태

```
ezis-postgres-rag   Up (healthy)   0.0.0.0:35432->5432/tcp
ezis-rag-app        Not started    (빌드 필요)
```

---

## 파일 위치

| 파일 | 경로 |
|------|------|
| 마이그레이션 계획 | MIGRATION_PLAN.md |
| Dockerfile | Dockerfile |
| docker-compose | docker-compose.yml |
| 환경변수 템플릿 | .env.example |
| DB 백업 | backup/ezis_rag_backup.dump |

---

## 주의사항

1. **API 키**: `.env` 파일에 GEMINI_API_KEY, OPENAI_API_KEY 설정 필요
2. **PostgreSQL 이미지**: `gzdaniel/postgres-for-rag:16.6` 사용 (pgvector + AGE 포함)
3. **빌드 시간**: mineru[core] 의존성으로 첫 빌드 시 10-15분 소요 (캐시 있으면 1-2분)
