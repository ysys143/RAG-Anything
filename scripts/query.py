"""
RAG-Anything Query - RAG 시스템 질의 (PostgreSQL Backend)

Usage:
    # 단일 질의
    python scripts/query.py "EZIS 시스템의 로그인 프로세스는?"

    # 쿼리 모드 지정
    python scripts/query.py "Oracle 연결 설정 방법" --mode local
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
import asyncio
import argparse
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# PostgreSQL configuration
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_USER", "pgvector")
os.environ.setdefault("POSTGRES_PASSWORD", "pgvector")
os.environ.setdefault("POSTGRES_DATABASE", "ezis_rag")

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, setup_logger

from raganything import RAGAnything, RAGAnythingConfig

# Setup logger
setup_logger("lightrag", level="INFO")

# Configuration
WORKING_DIR = os.getenv("WORKING_DIR", "./rag_storage")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))

# Default system prompt (페르소나)
DEFAULT_SYSTEM_PROMPT = """당신은 DBMS 모니터링 및 튜닝 전문기업 위데이터랩의 DBA 겸 고객지원 엔지니어입니다.
회사 제품에 대해 친절하고 전문적으로 안내합니다.
너무 일반적인 설명보다는 제품의 구체적인 특징, 장점, 실제 사용법을 중심으로 답변합니다.
답변은 한국어로 작성하며, 기술 용어는 필요시 영문을 병기합니다."""


async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> str:
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=OPENAI_API_KEY,
        **kwargs,
    )


async def vision_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    image_data=None,
    messages=None,
    **kwargs,
) -> str:
    if messages:
        return await openai_complete_if_cache(
            VISION_MODEL,
            "",
            system_prompt=None,
            history_messages=[],
            messages=messages,
            api_key=OPENAI_API_KEY,
            **kwargs,
        )
    elif image_data:
        return await openai_complete_if_cache(
            VISION_MODEL,
            "",
            system_prompt=None,
            history_messages=[],
            messages=[
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                    ],
                },
            ],
            api_key=OPENAI_API_KEY,
            **kwargs,
        )
    else:
        return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)


embedding_func = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM,
    max_token_size=8192,
    func=lambda texts: openai_embed(
        texts,
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    ),
)


async def create_rag_instance(use_vlm: bool = True) -> RAGAnything:
    """RAGAnything 인스턴스 생성 및 초기화"""
    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func if use_vlm else None,
        embedding_func=embedding_func,
        lightrag_kwargs={
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

    return rag


async def execute_query(
    rag: RAGAnything,
    query: str,
    mode: str = "mix",
    system_prompt: Optional[str] = None,
    vlm_enhanced: bool = True,
) -> str:
    """쿼리 실행"""
    effective_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    result = await rag.aquery(
        query=query,
        mode=mode,
        system_prompt=effective_prompt,
        vlm_enhanced=vlm_enhanced,
    )

    return result


async def interactive_mode(rag: RAGAnything, mode: str, system_prompt: Optional[str], vlm_enhanced: bool):
    """대화형 쿼리 모드"""
    print("\n" + "=" * 60)
    print("EZIS RAG 대화형 모드")
    print("=" * 60)
    print(f"쿼리 모드: {mode}")
    print(f"VLM 활성화: {vlm_enhanced}")
    print("\n명령어:")
    print("  /mode <local|global|hybrid|naive|mix>  - 쿼리 모드 변경")
    print("  /vlm                                    - VLM 토글")
    print("  /clear                                  - 화면 정리")
    print("  /help                                   - 도움말")
    print("  /quit 또는 /exit                        - 종료")
    print("-" * 60)

    current_mode = mode
    current_vlm = vlm_enhanced

    while True:
        try:
            user_input = input("\n질문> ").strip()
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
        print("\n검색 중...")
        try:
            result = await execute_query(
                rag=rag,
                query=user_input,
                mode=current_mode,
                system_prompt=system_prompt,
                vlm_enhanced=current_vlm,
            )
            print("\n" + "-" * 60)
            print(result)
            print("-" * 60)
        except Exception as e:
            print(f"오류 발생: {e}")


async def single_query(
    query: str,
    mode: str,
    system_prompt: Optional[str],
    vlm_enhanced: bool,
):
    """단일 쿼리 실행"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다")

    rag = await create_rag_instance(use_vlm=vlm_enhanced)

    try:
        result = await execute_query(
            rag=rag,
            query=query,
            mode=mode,
            system_prompt=system_prompt,
            vlm_enhanced=vlm_enhanced,
        )
        print(result)
    finally:
        try:
            await rag.finalize_storages()
        except Exception:
            pass


async def interactive_session(
    mode: str,
    system_prompt: Optional[str],
    vlm_enhanced: bool,
):
    """대화형 세션 실행"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다")

    rag = await create_rag_instance(use_vlm=vlm_enhanced)

    try:
        await interactive_mode(rag, mode, system_prompt, vlm_enhanced)
    finally:
        try:
            await rag.finalize_storages()
        except Exception:
            pass


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

    if args.interactive:
        asyncio.run(
            interactive_session(
                mode=args.mode,
                system_prompt=args.system_prompt,
                vlm_enhanced=not args.no_vlm,
            )
        )
    elif args.query:
        asyncio.run(
            single_query(
                query=args.query,
                mode=args.mode,
                system_prompt=args.system_prompt,
                vlm_enhanced=not args.no_vlm,
            )
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
