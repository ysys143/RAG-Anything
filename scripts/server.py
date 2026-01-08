"""
RAGAnything Server - 멀티모달 RAG API 서버 (PostgreSQL Backend)

Usage:
    python scripts/server.py
    python scripts/server.py --port 8000

Endpoints:
    POST /query         - 일반 쿼리
    POST /query/stream  - 스트리밍 쿼리
    GET  /health        - 헬스 체크
    GET  /storage/info  - 저장소 정보

Prerequisites:
    - PostgreSQL with pgvector and Apache AGE extensions
    - docker compose up -d
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Optional, Union, AsyncIterator

from dotenv import load_dotenv

load_dotenv()

# PostgreSQL configuration
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_USER", "pgvector")
os.environ.setdefault("POSTGRES_PASSWORD", "pgvector")
os.environ.setdefault("POSTGRES_DATABASE", "ezis_rag")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import base64

from google import genai
from google.genai import types

from lightrag.llm.gemini import gemini_model_complete
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

# Configuration from environment
WORKING_DIR = os.getenv("WORKING_DIR", "./rag_storage")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# OpenAI configuration (for embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))

# Global instances
rag: Optional[RAGAnything] = None


async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=GEMINI_API_KEY,
        model_name=GEMINI_MODEL,
        **kwargs,
    )


async def vision_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    image_data=None,
    messages=None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """Gemini vision model function supporting text and image inputs."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build content parts
    contents = []

    if messages:
        # Handle pre-built messages format (convert from OpenAI format)
        for msg in messages:
            if msg is None:
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                contents.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        contents.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        # Extract base64 from data URL
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            b64_data = url.split(",", 1)[-1]
                            image_bytes = base64.b64decode(b64_data)
                            contents.append(
                                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                            )
    elif image_data:
        # Handle direct base64 image data
        if system_prompt:
            contents.append(system_prompt)
        contents.append(prompt)
        image_bytes = base64.b64decode(image_data)
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
    else:
        # Text-only fallback
        return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # Call Gemini API
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=GEMINI_MODEL,
        contents=contents,
    )

    return response.text


embedding_func = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM,
    max_token_size=8192,
    func=lambda texts: openai_embed(
        texts,
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    ),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 RAGAnything 초기화/정리"""
    global rag

    print("Initializing RAGAnything server...")
    print(f"Working directory: {WORKING_DIR}")
    print(f"Gemini Model: {GEMINI_MODEL} (LLM + Vision)")

    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        parser=os.getenv("PARSER", "mineru"),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        parser_output_dir=OUTPUT_DIR,
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "llm_model_name": GEMINI_MODEL,
            "kv_storage": "PGKVStorage",
            "vector_storage": "PGVectorStorage",
            "graph_storage": "PGGraphStorage",
            "doc_status_storage": "PGDocStatusStorage",
        },
    )

    # LightRAG 초기화
    init_result = await rag._ensure_lightrag_initialized()
    if not init_result.get("success", False):
        raise RuntimeError(f"LightRAG 초기화 실패: {init_result.get('error', 'Unknown error')}")

    print("RAGAnything server ready!")

    yield

    print("Shutting down RAGAnything server...")
    if rag:
        await rag.finalize_storages()


app = FastAPI(
    title="RAGAnything Server",
    description="멀티모달 RAG API 서버",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (파싱된 이미지 등)
if os.path.exists(OUTPUT_DIR):
    app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    conversation_history: list = []
    history_turns: int = 0


class QueryResponse(BaseModel):
    response: str
    mode: str
    referenced_images: list[str] = []


@app.get("/health")
async def health():
    return {"status": "ok", "service": "raganything"}


@app.get("/storage/info")
async def storage_info():
    """저장소 백엔드 정보 조회"""
    return {
        "kv_storage": os.getenv("KV_STORAGE", "PGKVStorage"),
        "vector_storage": os.getenv("VECTOR_STORAGE", "PGVectorStorage"),
        "graph_storage": os.getenv("GRAPH_STORAGE", "PGGraphStorage"),
        "doc_status_storage": os.getenv("DOC_STATUS_STORAGE", "PGDocStatusStorage"),
        "postgres_host": os.getenv("POSTGRES_HOST", "localhost"),
        "postgres_database": os.getenv("POSTGRES_DATABASE", "ezis_rag"),
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """일반 쿼리 엔드포인트"""
    if not rag:
        raise HTTPException(status_code=503, detail="RAGAnything not initialized")

    try:
        # 멀티턴 컨텍스트 구성
        full_query = request.query
        if request.conversation_history and request.history_turns > 0:
            history = request.conversation_history[-(request.history_turns * 2):]
            context_parts = ["Previous conversation:"]
            for msg in history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")[:500]
                context_parts.append(f"{role}: {content}")
            context_parts.append(f"\nCurrent question: {request.query}")
            full_query = "\n".join(context_parts)

        result = await rag.aquery(full_query, mode=request.mode)

        # VLM 활성화 시 dict 반환, 아니면 string 반환
        if isinstance(result, dict):
            return QueryResponse(
                response=result["response"],
                mode=request.mode,
                referenced_images=result.get("referenced_images", []),
            )
        return QueryResponse(response=result, mode=request.mode)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """스트리밍 쿼리 엔드포인트 (lightrag-server 호환)"""
    if not rag:
        raise HTTPException(status_code=503, detail="RAGAnything not initialized")

    async def generate():
        try:
            # 멀티턴 컨텍스트 구성
            full_query = request.query
            if request.conversation_history and request.history_turns > 0:
                history = request.conversation_history[-(request.history_turns * 2):]
                context_parts = ["Previous conversation:"]
                for msg in history:
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    content = msg.get("content", "")[:500]
                    context_parts.append(f"{role}: {content}")
                context_parts.append(f"\nCurrent question: {request.query}")
                full_query = "\n".join(context_parts)

            # RAGAnything 쿼리 실행 (논스트리밍)
            result = await rag.aquery(full_query, mode=request.mode)

            # VLM 활성화 시 dict 반환 처리
            response_text = result["response"] if isinstance(result, dict) else result
            referenced_images = result.get("referenced_images", []) if isinstance(result, dict) else []

            # lightrag-server 형식으로 스트리밍 출력
            # 청크 단위로 분할해서 전송
            chunk_size = 10
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                yield json.dumps({"response": chunk}) + "\n"
                await asyncio.sleep(0.01)

            # 마지막에 참조 이미지 정보 전송
            if referenced_images:
                yield json.dumps({"referenced_images": referenced_images}) + "\n"

        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="RAGAnything Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=9621, help="Port")
    args = parser.parse_args()

    print(f"Starting RAGAnything Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
