"""
RAG Query Benchmark - 쿼리 모드별 응답 시간 측정

Usage:
    python scripts/benchmark_query.py "질문 내용"
    python scripts/benchmark_query.py "질문 내용" --runs 3
    python scripts/benchmark_query.py "질문 내용" --modes naive hybrid
    python scripts/benchmark_query.py --all-modes  # 모든 모드 테스트 (기본 질문)
    python scripts/benchmark_query.py "질문" --save  # 결과 파일 저장

Example:
    python scripts/benchmark_query.py "EZIS 시스템의 로그인 프로세스는?"
    python scripts/benchmark_query.py "시스템 아키텍처 설명" --runs 5 --save
"""

import os
import sys
import time
import asyncio
import argparse
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

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
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig

# Configuration
WORKING_DIR = os.getenv("WORKING_DIR", "./rag_storage")

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# OpenAI configuration (for embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

QUERY_MODES = ["naive", "local", "global", "hybrid", "mix"]
DEFAULT_QUERY = "시스템 아키텍처 설명"


@dataclass
class BenchmarkResult:
    mode: str
    times: list[float] = field(default_factory=list)
    response_length: int = 0
    response_text: str = ""
    success: bool = True
    error: Optional[str] = None

    @property
    def avg_time(self) -> float:
        return statistics.mean(self.times) if self.times else 0

    @property
    def min_time(self) -> float:
        return min(self.times) if self.times else 0

    @property
    def max_time(self) -> float:
        return max(self.times) if self.times else 0

    @property
    def std_dev(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=GEMINI_API_KEY,
        model_name=GEMINI_MODEL,
        **kwargs,
    )


async def vision_model_func(
    prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
):
    client = genai.Client(api_key=GEMINI_API_KEY)
    contents = []

    if messages:
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
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            b64_data = url.split(",", 1)[-1]
                            image_bytes = base64.b64decode(b64_data)
                            contents.append(
                                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                            )
    elif image_data:
        if system_prompt:
            contents.append(system_prompt)
        contents.append(prompt)
        image_bytes = base64.b64decode(image_data)
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
    else:
        return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)

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


async def create_rag() -> RAGAnything:
    config = RAGAnythingConfig(working_dir=WORKING_DIR)

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

    init_result = await rag._ensure_lightrag_initialized()
    if not init_result.get("success", False):
        raise RuntimeError(f"LightRAG 초기화 실패: {init_result.get('error')}")

    return rag


async def benchmark_query(
    rag: RAGAnything,
    query: str,
    mode: str,
    runs: int = 1,
) -> BenchmarkResult:
    times = []
    response_length = 0
    response_text = ""
    error = None

    for i in range(runs):
        try:
            start = time.perf_counter()
            result = await rag.aquery(query, mode=mode, vlm_enhanced=False)
            elapsed = time.perf_counter() - start

            times.append(elapsed)

            if isinstance(result, dict):
                response_text = result.get("response", "")
            else:
                response_text = result or ""

            response_length = len(response_text)

        except Exception as e:
            error = str(e)
            break

    return BenchmarkResult(
        mode=mode,
        times=times,
        response_length=response_length,
        response_text=response_text,
        success=error is None,
        error=error,
    )


def print_results(results: list[BenchmarkResult], query: str, runs: int):
    print("\n" + "=" * 70)
    print(f"Query Benchmark Results (Gemini: {GEMINI_MODEL})")
    print("=" * 70)
    print(f"Query: {query[:50]}{'...' if len(query) > 50 else ''}")
    print(f"Runs per mode: {runs}")
    print("-" * 70)
    print(f"{'Mode':<10} {'Avg (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'StdDev':<10} {'Resp Len':<10}")
    print("-" * 70)

    for r in results:
        if r.success:
            print(
                f"{r.mode:<10} "
                f"{r.avg_time:<10.3f} "
                f"{r.min_time:<10.3f} "
                f"{r.max_time:<10.3f} "
                f"{r.std_dev:<10.3f} "
                f"{r.response_length:<10}"
            )
        else:
            print(f"{r.mode:<10} FAILED: {r.error}")

    print("-" * 70)

    # Summary
    successful = [r for r in results if r.success]
    if successful:
        fastest = min(successful, key=lambda x: x.avg_time)
        slowest = max(successful, key=lambda x: x.avg_time)
        print(f"\nFastest: {fastest.mode} ({fastest.avg_time:.3f}s)")
        print(f"Slowest: {slowest.mode} ({slowest.avg_time:.3f}s)")
        print(f"Difference: {slowest.avg_time - fastest.avg_time:.3f}s ({slowest.avg_time / fastest.avg_time:.1f}x)")


def save_results_to_markdown(results: list[BenchmarkResult], query: str, runs: int) -> str:
    """결과를 마크다운 파일로 저장"""
    # 결과 디렉토리 생성
    results_dir = Path("docs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 파일명 생성
    date_str = datetime.now().strftime("%y%m%d")
    filename = results_dir / f"LATENCY_BY_TYPE_{date_str}.md"

    # 기존 파일이 있으면 번호 추가
    counter = 1
    while filename.exists():
        filename = results_dir / f"LATENCY_BY_TYPE_{date_str}_{counter}.md"
        counter += 1

    # 마크다운 내용 생성
    lines = [
        f"# Query Benchmark Results",
        f"",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Model**: {GEMINI_MODEL}",
        f"- **Runs per mode**: {runs}",
        f"",
        f"## Query",
        f"",
        f"```",
        f"{query}",
        f"```",
        f"",
        f"## Latency Summary",
        f"",
        f"| Mode | Avg (s) | Min (s) | Max (s) | StdDev | Resp Len |",
        f"|------|---------|---------|---------|--------|----------|",
    ]

    successful = []
    for r in results:
        if r.success:
            lines.append(
                f"| {r.mode} | {r.avg_time:.3f} | {r.min_time:.3f} | {r.max_time:.3f} | {r.std_dev:.3f} | {r.response_length} |"
            )
            successful.append(r)
        else:
            lines.append(f"| {r.mode} | FAILED | - | - | - | - |")

    if successful:
        fastest = min(successful, key=lambda x: x.avg_time)
        slowest = max(successful, key=lambda x: x.avg_time)
        lines.extend([
            f"",
            f"**Fastest**: {fastest.mode} ({fastest.avg_time:.3f}s)",
            f"",
            f"**Slowest**: {slowest.mode} ({slowest.avg_time:.3f}s)",
            f"",
            f"**Difference**: {slowest.avg_time - fastest.avg_time:.3f}s ({slowest.avg_time / fastest.avg_time:.1f}x)",
        ])

    # 각 모드별 응답 추가
    lines.extend([
        f"",
        f"---",
        f"",
        f"## Responses by Mode",
    ])

    for r in results:
        lines.extend([
            f"",
            f"### {r.mode.upper()}",
            f"",
        ])
        if r.success:
            lines.extend([
                f"**Time**: {r.avg_time:.3f}s | **Length**: {r.response_length} chars",
                f"",
                f"<details>",
                f"<summary>Response</summary>",
                f"",
                f"{r.response_text}",
                f"",
                f"</details>",
            ])
        else:
            lines.append(f"**Error**: {r.error}")

    # 파일 저장
    content = "\n".join(lines)
    filename.write_text(content, encoding="utf-8")

    return str(filename)


async def run_benchmark(query: str, modes: list[str], runs: int, save: bool = False):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다")

    print(f"Initializing RAG system...")
    rag = await create_rag()

    results = []
    try:
        for mode in modes:
            print(f"Benchmarking mode: {mode} ({runs} runs)...")
            result = await benchmark_query(rag, query, mode, runs)
            results.append(result)

        print_results(results, query, runs)

        if save:
            filepath = save_results_to_markdown(results, query, runs)
            print(f"\nResults saved to: {filepath}")

    finally:
        await rag.finalize_storages()


def main():
    parser = argparse.ArgumentParser(
        description="RAG Query Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=DEFAULT_QUERY,
        help=f"쿼리 내용 (기본: '{DEFAULT_QUERY}')",
    )
    parser.add_argument(
        "--modes",
        "-m",
        nargs="+",
        choices=QUERY_MODES,
        default=None,
        help="테스트할 모드 (기본: 모든 모드)",
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=1,
        help="모드당 실행 횟수 (기본: 1)",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="모든 모드 테스트",
    )
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="결과를 docs/results/LATENCY_BY_TYPE_{yymmdd}.md 로 저장",
    )

    args = parser.parse_args()

    modes = args.modes if args.modes else QUERY_MODES

    try:
        asyncio.run(run_benchmark(args.query, modes, args.runs, args.save))
    except KeyboardInterrupt:
        print("\n중단됨.")
        sys.exit(1)


if __name__ == "__main__":
    main()
