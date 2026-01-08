"""
RAG-Anything Query - RAG 시스템 질의 (PostgreSQL Backend)

Usage:
    # 단일 질의
    python scripts/query.py "EZIS 시스템의 로그인 프로세스는?"

    # 스트리밍 응답
    python scripts/query.py "Oracle 연결 설정 방법" --stream

    # 쿼리 모드 지정
    python scripts/query.py "시스템 아키텍처 설명" --mode global

    # 대화형 모드
    python scripts/query.py --interactive

    # VLM 비활성화 (텍스트 전용)
    python scripts/query.py "질문" --no-vlm

Prerequisites:
    - PostgreSQL with pgvector and Apache AGE extensions
    - docker compose up -d
    - 문서가 이미 인덱싱되어 있어야 함 (scripts/ingest.py 실행)
"""

import os
import sys
import signal
import asyncio
import argparse
from typing import Optional, AsyncIterator, Union
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

# PostgreSQL configuration
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_USER", "pgvector")
os.environ.setdefault("POSTGRES_PASSWORD", "pgvector")
os.environ.setdefault("POSTGRES_DATABASE", "ezis_rag")

import base64

from google import genai
from google.genai import types

from lightrag.llm.gemini import gemini_model_complete
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc, setup_logger

from raganything import RAGAnything, RAGAnythingConfig

# Setup logger
setup_logger("lightrag", level="INFO")

# Configuration
WORKING_DIR = os.getenv("WORKING_DIR", "./rag_storage")

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# OpenAI configuration (for embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# Default system prompt (페르소나)
DEFAULT_SYSTEM_PROMPT = """당신은 DBMS 모니터링 및 튜닝 전문기업 위데이터랩의 DBA 겸 고객지원 엔지니어입니다.
회사 제품에 대해 친절하고 전문적으로 안내합니다.
너무 일반적인 설명보다는 제품의 구체적인 특징, 장점, 실제 사용법을 중심으로 답변합니다.
답변은 한국어로 작성하며, 기술 용어는 필요시 영문을 병기합니다."""


# Graceful shutdown handling
class GracefulShutdown:
    """Graceful shutdown handler for async operations"""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self._rag: Optional[RAGAnything] = None

    def register_rag(self, rag: RAGAnything):
        """Register RAG instance for cleanup"""
        self._rag = rag

    async def cleanup(self):
        """Cleanup resources"""
        if self._rag:
            try:
                await self._rag.finalize_storages()
            except Exception:
                pass

    def trigger_shutdown(self):
        """Trigger shutdown event"""
        self.shutdown_event.set()


# Global shutdown handler
_shutdown_handler = GracefulShutdown()


def setup_signal_handlers(loop: asyncio.AbstractEventLoop):
    """Setup signal handlers for graceful shutdown"""

    def signal_handler(sig, frame):
        print("\n\n중단 요청 수신. 정리 중...")
        _shutdown_handler.trigger_shutdown()
        # Schedule cleanup
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(_shutdown_handler.cleanup())
        )

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


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
async def create_rag_context(use_vlm: bool = True):
    """RAGAnything 컨텍스트 매니저 (자동 정리)"""
    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func if use_vlm else None,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "llm_model_name": GEMINI_MODEL,
            "kv_storage": "PGKVStorage",
            "vector_storage": "PGVectorStorage",
            "graph_storage": "PGGraphStorage",
            "doc_status_storage": "PGDocStatusStorage",
        },
    )

    # Register for graceful shutdown
    _shutdown_handler.register_rag(rag)

    # LightRAG 초기화
    init_result = await rag._ensure_lightrag_initialized()
    if not init_result.get("success", False):
        raise RuntimeError(f"LightRAG 초기화 실패: {init_result.get('error', 'Unknown error')}")

    try:
        yield rag
    finally:
        try:
            await rag.finalize_storages()
        except Exception:
            pass


async def execute_query(
    rag: RAGAnything,
    query: str,
    mode: str = "mix",
    system_prompt: Optional[str] = None,
    vlm_enhanced: bool = True,
    stream: bool = False,
) -> Union[str, AsyncIterator[str], dict]:
    """쿼리 실행

    Returns:
        str: 기본 응답 (VLM 비활성화 시)
        AsyncIterator: stream=True
        dict: VLM 활성화 시 {"response": str, "referenced_images": list[str]}
    """
    effective_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    result = await rag.aquery(
        query=query,
        mode=mode,
        system_prompt=effective_prompt,
        vlm_enhanced=vlm_enhanced,
        stream=stream,
    )

    return result


async def print_streaming_response(response_iterator: AsyncIterator[str]):
    """스트리밍 응답 출력"""
    try:
        async for chunk in response_iterator:
            if _shutdown_handler.shutdown_event.is_set():
                print("\n[중단됨]")
                break
            print(chunk, end="", flush=True)
        print()  # 마지막 줄바꿈
    except asyncio.CancelledError:
        print("\n[취소됨]")


async def interactive_mode(
    rag: RAGAnything,
    mode: str,
    system_prompt: Optional[str],
    vlm_enhanced: bool,
    stream: bool,
):
    """대화형 쿼리 모드"""
    print("\n" + "=" * 60)
    print("위데이터랩 RAG 대화형 모드")
    print("=" * 60)
    print(f"쿼리 모드: {mode}")
    print(f"VLM 활성화: {vlm_enhanced}")
    print(f"스트리밍: {stream}")
    print("\n명령어:")
    print("  /mode <local|global|hybrid|naive|mix>  - 쿼리 모드 변경")
    print("  /vlm                                    - VLM 토글")
    print("  /stream                                 - 스트리밍 토글")
    print("  /clear                                  - 화면 정리")
    print("  /help                                   - 도움말")
    print("  /quit 또는 /exit                        - 종료")
    print("-" * 60)

    current_mode = mode
    current_vlm = vlm_enhanced
    current_stream = stream

    while not _shutdown_handler.shutdown_event.is_set():
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("\n질문> ").strip()
            )
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not user_input:
            continue

        # 명령어 처리
        if user_input.startswith("/"):
            cmd_parts = user_input[1:].split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else None

            if cmd in ("quit", "exit", "q"):
                print("종료합니다.")
                break
            elif cmd == "mode" and cmd_arg:
                if cmd_arg in ("local", "global", "hybrid", "naive", "mix"):
                    current_mode = cmd_arg
                    print(f"쿼리 모드 변경: {current_mode}")
                else:
                    print("유효하지 않은 모드입니다. (local, global, hybrid, naive, mix)")
            elif cmd == "vlm":
                current_vlm = not current_vlm
                print(f"VLM {'활성화' if current_vlm else '비활성화'}")
            elif cmd == "stream":
                current_stream = not current_stream
                print(f"스트리밍 {'활성화' if current_stream else '비활성화'}")
            elif cmd == "clear":
                os.system("clear" if os.name != "nt" else "cls")
            elif cmd == "help":
                print("\n쿼리 모드 설명:")
                print("  local   - 로컬 컨텍스트 기반 (특정 엔티티/관계)")
                print("  global  - 전역 컨텍스트 기반 (커뮤니티 요약)")
                print("  hybrid  - local + global 결합")
                print("  naive   - 단순 벡터 검색")
                print("  mix     - 모든 모드 결합 (기본값)")
            else:
                print(f"알 수 없는 명령어: {cmd}")
            continue

        # 쿼리 실행
        if not current_stream:
            print("\n검색 중...")

        try:
            result = await execute_query(
                rag=rag,
                query=user_input,
                mode=current_mode,
                system_prompt=system_prompt,
                vlm_enhanced=current_vlm,
                stream=current_stream,
            )

            print("\n" + "-" * 60)
            if current_stream and hasattr(result, "__aiter__"):
                await print_streaming_response(result)
            else:
                print(result)
            print("-" * 60)
        except asyncio.CancelledError:
            print("\n[쿼리 취소됨]")
        except Exception as e:
            print(f"오류 발생: {e}")


async def single_query(
    query: str,
    mode: str,
    system_prompt: Optional[str],
    vlm_enhanced: bool,
    stream: bool,
):
    """단일 쿼리 실행"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다")

    async with create_rag_context(use_vlm=vlm_enhanced) as rag:
        result = await execute_query(
            rag=rag,
            query=query,
            mode=mode,
            system_prompt=system_prompt,
            vlm_enhanced=vlm_enhanced,
            stream=stream,
        )

        if stream and hasattr(result, "__aiter__"):
            await print_streaming_response(result)
        elif isinstance(result, dict):
            # VLM 응답: 참조 이미지와 함께 출력
            print(result["response"])
            if result.get("referenced_images"):
                print("\n" + "-" * 40)
                print("참조된 이미지:")
                for img_path in result["referenced_images"]:
                    print(f"  - {img_path}")
        else:
            print(result)


async def interactive_session(
    mode: str,
    system_prompt: Optional[str],
    vlm_enhanced: bool,
    stream: bool,
):
    """대화형 세션 실행"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다")

    async with create_rag_context(use_vlm=vlm_enhanced) as rag:
        await interactive_mode(rag, mode, system_prompt, vlm_enhanced, stream)


async def cancel_all_tasks(loop: asyncio.AbstractEventLoop):
    """Cancel all pending tasks gracefully"""
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]

    if not tasks:
        return

    for task in tasks:
        task.cancel()

    # Wait for all tasks to be cancelled with timeout
    await asyncio.gather(*tasks, return_exceptions=True)

    # Give tasks a moment to clean up
    await asyncio.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(
        description="RAG-Anything Query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="질의할 내용 (--interactive 모드에서는 생략)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="mix",
        choices=["local", "global", "hybrid", "naive", "mix"],
        help="쿼리 모드 (기본: mix)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="대화형 모드 실행",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="스트리밍 응답 활성화",
    )
    parser.add_argument(
        "--system-prompt",
        "-s",
        type=str,
        default=None,
        help="커스텀 시스템 프롬프트 (페르소나)",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="VLM(Vision Language Model) 비활성화",
    )

    args = parser.parse_args()

    # Create event loop and setup signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    setup_signal_handlers(loop)

    try:
        if args.interactive:
            loop.run_until_complete(
                interactive_session(
                    mode=args.mode,
                    system_prompt=args.system_prompt,
                    vlm_enhanced=not args.no_vlm,
                    stream=args.stream,
                )
            )
        elif args.query:
            loop.run_until_complete(
                single_query(
                    query=args.query,
                    mode=args.mode,
                    system_prompt=args.system_prompt,
                    vlm_enhanced=not args.no_vlm,
                    stream=args.stream,
                )
            )
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        # Cleanup: finalize storages first
        try:
            loop.run_until_complete(_shutdown_handler.cleanup())
        except Exception:
            pass

        # Cancel all remaining tasks to prevent "Task was destroyed" warnings
        try:
            loop.run_until_complete(cancel_all_tasks(loop))
        except Exception:
            pass

        # Shutdown async generators
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass

        loop.close()


if __name__ == "__main__":
    main()
