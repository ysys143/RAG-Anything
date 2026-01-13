# API Reference

## Overview

EZIS RAG는 두 가지 API 인터페이스를 제공한다:
1. **HTTP API** (`scripts/server.py`): FastAPI 기반 REST API
2. **Python API** (`raganything`): 직접 호출 가능한 비동기 메서드

---

## HTTP API

### Base URL

```
http://localhost:9621
```

### Endpoints

#### GET /health

헬스 체크 엔드포인트.

**Response:**
```json
{
  "status": "ok",
  "service": "raganything"
}
```

---

#### GET /storage/info

스토리지 백엔드 정보 조회.

**Response:**
```json
{
  "kv_storage": "PGKVStorage",
  "vector_storage": "PGVectorStorage",
  "graph_storage": "PGGraphStorage",
  "doc_status_storage": "PGDocStatusStorage",
  "postgres_host": "localhost",
  "postgres_database": "ezis_rag"
}
```

---

#### POST /query

일반 쿼리 엔드포인트.

**Request Body:**
```json
{
  "query": "EZIS 시스템 소개",
  "mode": "hybrid",
  "conversation_history": [],
  "history_turns": 0
}
```

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `query` | string | (필수) | 질의 텍스트 |
| `mode` | string | `"hybrid"` | 쿼리 모드 (local/global/hybrid/naive/mix) |
| `conversation_history` | array | `[]` | 이전 대화 히스토리 |
| `history_turns` | int | `0` | 포함할 대화 턴 수 |

**Response:**
```json
{
  "response": "EZIS는 데이터베이스 모니터링 및 튜닝 솔루션으로...",
  "mode": "hybrid",
  "referenced_images": [
    "/path/to/image1.png",
    "/path/to/image2.jpg"
  ]
}
```

---

#### POST /query/stream

스트리밍 쿼리 엔드포인트. NDJSON 형식으로 응답을 스트리밍한다.

**Request Body:** `/query`와 동일

**Response (NDJSON Stream):**
```
{"response": "EZIS는 "}
{"response": "데이터베이스 "}
{"response": "모니터링 "}
...
{"referenced_images": ["/path/to/image.png"]}
```

**Content-Type:** `application/x-ndjson`

---

## Python API

### RAGAnything Class

#### Constructor

```python
from raganything import RAGAnything, RAGAnythingConfig

rag = RAGAnything(
    config=RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    ),
    llm_model_func=llm_func,          # async callable
    vision_model_func=vision_func,     # async callable (optional)
    embedding_func=embedding_func,     # EmbeddingFunc
    lightrag_kwargs={
        "kv_storage": "PGKVStorage",
        "vector_storage": "PGVectorStorage",
        "graph_storage": "PGGraphStorage",
        "doc_status_storage": "PGDocStatusStorage",
    },
)
```

---

### Query Methods

#### aquery()

텍스트 쿼리. `vision_model_func`이 설정되면 자동으로 VLM 모드를 활성화한다.

```python
async def aquery(
    self,
    query: str,
    mode: str = "mix",
    system_prompt: str | None = None,
    **kwargs
) -> str | dict | AsyncIterator[str]
```

**Parameters:**
| 이름 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `query` | str | (필수) | 질의 텍스트 |
| `mode` | str | `"mix"` | 쿼리 모드 |
| `system_prompt` | str | `None` | 시스템 프롬프트 |
| `vlm_enhanced` | bool | `auto` | VLM 활성화 여부 |
| `stream` | bool | `False` | 스트리밍 모드 |
| `return_images` | bool | `True` | 참조 이미지 반환 (VLM 모드) |

**Returns:**
- `str`: 기본 응답
- `dict`: VLM 모드 + `return_images=True` → `{"response": str, "referenced_images": list[str]}`
- `AsyncIterator[str]`: `stream=True`

**Example:**
```python
# 기본 쿼리
result = await rag.aquery("EZIS 시스템 소개", mode="hybrid")
print(result)

# VLM + 참조 이미지
result = await rag.aquery("화면 구성 설명", return_images=True)
print(result["response"])
print(result["referenced_images"])

# 스트리밍
async for chunk in await rag.aquery("설치 방법", stream=True):
    print(chunk, end="")
```

---

#### aquery_with_multimodal()

멀티모달 콘텐츠를 포함한 쿼리.

```python
async def aquery_with_multimodal(
    self,
    query: str,
    multimodal_content: List[Dict[str, Any]] = None,
    mode: str = "mix",
    **kwargs
) -> str
```

**multimodal_content 형식:**
```python
# 이미지
{"type": "image", "img_path": "/path/to/image.jpg"}

# 테이블
{"type": "table", "table_data": "Name,Age\nAlice,25\nBob,30", "table_caption": "사용자 목록"}

# 수식
{"type": "equation", "latex": "E = mc^2", "equation_caption": "질량-에너지 등가"}
```

**Example:**
```python
result = await rag.aquery_with_multimodal(
    query="이 테이블의 데이터 트렌드 분석",
    multimodal_content=[{
        "type": "table",
        "table_data": "Month,Revenue\nJan,100\nFeb,150\nMar,200",
        "table_caption": "분기별 매출"
    }],
    mode="hybrid"
)
```

---

#### aquery_vlm_enhanced()

VLM 강화 쿼리. 검색된 컨텍스트의 이미지 경로를 base64로 변환하여 VLM에 전달한다.

```python
async def aquery_vlm_enhanced(
    self,
    query: str,
    mode: str = "mix",
    system_prompt: str | None = None,
    **kwargs
) -> str | dict | AsyncIterator[str]
```

**Internal Flow:**
1. `QueryParam(only_need_prompt=True)`로 컨텍스트 추출
2. 이미지 경로 패턴 매칭 (`Image Path: /path/to/image.jpg`)
3. 이미지를 base64로 인코딩, `[VLM_IMAGE_N]` 마커 삽입
4. VLM에 멀티모달 메시지 전송
5. `[REFERENCED_IMAGES: 1, 3]` 태그 파싱

---

### Document Processing Methods

#### process_document_complete()

문서 파싱부터 인덱싱까지 전체 파이프라인 실행.

```python
async def process_document_complete(
    self,
    file_path: str,
    output_dir: str = None,
    parse_method: str = None,
    **kwargs
) -> dict
```

**Returns:**
```python
{
    "doc_id": "doc-abc123...",
    "file_path": "/path/to/document.pdf",
    "status": "completed",
    "text_chunks": 42,
    "multimodal_items": 8
}
```

---

#### parse_document()

문서 파싱만 수행. 캐싱을 지원한다.

```python
async def parse_document(
    self,
    file_path: str,
    output_dir: str = None,
    parse_method: str = None,
    display_stats: bool = None,
    **kwargs
) -> tuple[List[Dict[str, Any]], str]
```

**Returns:** `(content_list, doc_id)`

---

#### get_document_processing_status()

문서 처리 상태 조회.

```python
async def get_document_processing_status(self, doc_id: str) -> dict
```

**Returns:**
```python
{
    "doc_id": "doc-abc123...",
    "status": "processed",
    "fully_processed": True,
    "multimodal_processed": True
}
```

---

### Storage Management

#### finalize_storages()

모든 스토리지 리소스 정리. 서버 종료 시 호출해야 한다.

```python
async def finalize_storages(self) -> None
```

**Example:**
```python
try:
    rag = RAGAnything(...)
    await rag._ensure_lightrag_initialized()
    result = await rag.aquery("질문")
finally:
    await rag.finalize_storages()
```

---

## Content List Format

MinerU 파서가 생성하는 콘텐츠 리스트 형식:

```python
# 텍스트
{
    "type": "text",
    "text": "내용...",
    "text_level": 0,      # 0: 본문, 1-6: 헤더 레벨
    "page_idx": 0
}

# 이미지
{
    "type": "image",
    "img_path": "/absolute/path/to/image.jpg",
    "image_caption": ["캡션 1", "캡션 2"],
    "image_footnote": ["주석"],
    "page_idx": 1
}

# 테이블
{
    "type": "table",
    "table_body": "| Header | Header |\n|--------|--------|",
    "table_caption": ["테이블 제목"],
    "table_footnote": [],
    "page_idx": 2
}

# 수식
{
    "type": "equation",
    "text": "E = mc^2",
    "text_format": "latex",
    "page_idx": 3
}
```

---

## Error Handling

### Common Exceptions

| Exception | 원인 | 해결 |
|-----------|------|------|
| `ValueError("No LightRAG instance")` | 문서 처리 전 쿼리 시도 | `process_document_complete()` 먼저 호출 |
| `ValueError("llm_model_func must be provided")` | LLM 함수 누락 | 생성자에 `llm_model_func` 전달 |
| `RuntimeError("Parser not installed")` | MinerU/Docling 미설치 | `pip install mineru` 또는 `pip install docling` |
| `asyncpg.exceptions.ConnectionFailure` | PostgreSQL 연결 실패 | Docker 컨테이너 상태 확인 |

### Graceful Shutdown

스크립트에서 SIGINT/SIGTERM 핸들링:

```python
import signal
import asyncio

shutdown_event = asyncio.Event()

def signal_handler(sig, frame):
    print("Shutting down...")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 쿼리 루프에서 체크
while not shutdown_event.is_set():
    result = await rag.aquery(query)
    ...

# 정리
await rag.finalize_storages()
```

---

## Rate Limits & Performance

### LLM 호출 제한

| 모델 | RPM | TPM |
|------|-----|-----|
| Gemini 2.5 Flash | 60 | 100K |
| OpenAI GPT-4.1 | 60 | 150K |
| text-embedding-3-small | 500 | 1M |

### 권장 설정

```python
lightrag_kwargs={
    "embedding_batch_num": 32,        # 임베딩 배치 크기
    "embedding_func_max_async": 16,   # 동시 임베딩 호출
    "llm_model_max_async": 4,         # 동시 LLM 호출
    "max_parallel_insert": 2,         # 동시 문서 삽입
}
```

---

*Last Updated: 2026-01-13*
