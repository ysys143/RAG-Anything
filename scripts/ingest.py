"""
RAG-Anything Document Ingest - PDF 문서 인덱싱 (PostgreSQL Backend)

Usage:
    python scripts/ingest.py manuals/EZIS_Oracle_Manual_20251216.pdf
    python scripts/ingest.py manuals/  # 폴더 내 모든 PDF
    python scripts/ingest.py manuals/ --force  # 강제 재처리

Prerequisites:
    - PostgreSQL with pgvector and Apache AGE extensions
    - docker compose up -d
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

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
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
SUPPORTED_EXTENSIONS = {".pdf"}

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


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


import hashlib
import json

# 파일 해시 -> doc_id 매핑 캐시 (로컬 JSON)
FILE_HASH_CACHE_PATH = Path(WORKING_DIR) / ".file_hash_cache.json"


def get_file_hash(file_path: Path) -> str:
    """파일 내용의 MD5 해시 계산"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_file_hash_cache() -> dict:
    """파일 해시 캐시 로드"""
    if FILE_HASH_CACHE_PATH.exists():
        try:
            with open(FILE_HASH_CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_file_hash_cache(cache: dict):
    """파일 해시 캐시 저장"""
    FILE_HASH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FILE_HASH_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def collect_pdf_files(path: Path, exclude_patterns: list[str] = None) -> list[Path]:
    """지정된 경로에서 PDF 파일 목록 수집

    Args:
        path: 파일 또는 폴더 경로
        exclude_patterns: 제외할 폴더/파일 패턴 (예: ["backup", "old"])
    """
    exclude_patterns = exclude_patterns or []

    def should_exclude(file_path: Path) -> bool:
        """파일 경로가 제외 패턴과 일치하는지 확인"""
        path_str = str(file_path).lower()
        return any(pattern.lower() in path_str for pattern in exclude_patterns)

    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            if should_exclude(path):
                print(f"제외됨 (패턴 일치): {path}")
                return []
            return [path]
        else:
            print(f"지원하지 않는 파일 형식: {path.suffix}")
            return []

    if path.is_dir():
        pdf_files = sorted(path.glob("**/*.pdf"))
        filtered = [f for f in pdf_files if not should_exclude(f)]
        excluded_count = len(pdf_files) - len(filtered)
        if excluded_count > 0:
            print(f"제외된 파일: {excluded_count}개 (패턴: {', '.join(exclude_patterns)})")
        return filtered

    return []


async def query_doc_status_by_filename(rag: RAGAnything, file_name: str) -> dict | None:
    """DB에서 파일명으로 doc_status 조회 (파싱 없이)"""
    try:
        if not hasattr(rag, 'lightrag') or rag.lightrag is None:
            return None

        db = rag.lightrag.doc_status.db
        workspace = rag.lightrag.workspace or "default"

        # db.query() 사용 (params는 list로 전달)
        sql = """
            SELECT id, status, metadata
            FROM lightrag_doc_status
            WHERE file_path = $1 AND workspace = $2
            LIMIT 1
        """
        result = await db.query(sql, params=[file_name, workspace])
        if result:
            # query()는 dict 반환: {"id": ..., "status": ..., "metadata": ...}
            metadata = result.get("metadata") or {}
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata) if metadata else {}
                except:
                    metadata = {}
            return {
                "doc_id": result.get("id"),
                "status": result.get("status"),
                "multimodal_processed": metadata.get("multimodal_processed", False) if isinstance(metadata, dict) else False,
            }
        return None
    except Exception as e:
        print(f"  DB 조회 오류: {e}")
        return None


async def check_document_fully_processed(
    rag: RAGAnything, file_path: Path, file_hash_cache: dict
) -> tuple[bool, str | None]:
    """
    문서가 완전히 처리되었는지 확인 (파싱 없이)

    1차: 파일 해시 캐시에서 doc_id 조회
    2차: DB에서 파일명으로 직접 조회

    파싱은 하지 않음 (새 문서는 False 반환)

    Returns:
        tuple[bool, str | None]: (완전 처리 여부, doc_id 또는 None)
    """
    try:
        file_hash = get_file_hash(file_path)
        doc_id = None

        # 1차: 파일 해시 캐시에서 조회
        cached = file_hash_cache.get(file_hash)
        if cached:
            doc_id = cached.get("doc_id")
            # 캐시에서 찾으면 doc_status 확인
            status = await rag.get_document_processing_status(doc_id)
            if status.get("fully_processed", False):
                return True, doc_id

        # 2차: DB에서 파일명으로 직접 조회 (파싱 없이)
        db_result = await query_doc_status_by_filename(rag, file_path.name)
        if db_result:
            doc_id = db_result["doc_id"]
            # 캐시 업데이트
            update_file_hash_cache(file_hash_cache, file_path, doc_id)

            # 텍스트 + 멀티모달 모두 완료된 경우에만 스킵
            text_done = db_result.get("status") == "processed"
            multimodal_done = db_result.get("multimodal_processed", False)

            if text_done and multimodal_done:
                return True, doc_id

        return False, doc_id

    except Exception as e:
        print(f"  상태 확인 중 오류: {e}")
        return False, None


def update_file_hash_cache(file_hash_cache: dict, file_path: Path, doc_id: str):
    """파일 해시 캐시 업데이트"""
    file_hash = get_file_hash(file_path)
    file_hash_cache[file_hash] = {
        "doc_id": doc_id,
        "file_path": str(file_path),
        "file_name": file_path.name,
    }


async def ingest_documents(
    files: list[Path],
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """문서들을 RAG 시스템에 인덱싱"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다")

    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        parser="mineru",
        parse_method="auto",
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
            "kv_storage": "PGKVStorage",
            "vector_storage": "PGVectorStorage",
            "graph_storage": "PGGraphStorage",
            "doc_status_storage": "PGDocStatusStorage",
        },
    )

    results = {
        "processed": [],
        "skipped": [],
        "failed": [],
    }

    # 파일 해시 캐시 로드
    file_hash_cache = load_file_hash_cache()

    try:
        # LightRAG 초기화 (중복 체크를 위해 필요)
        init_result = await rag._ensure_lightrag_initialized()
        if not init_result.get("success", False):
            raise RuntimeError(f"LightRAG 초기화 실패: {init_result.get('error', 'Unknown error')}")

        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {file_path.name}")

            if not force:
                # 파일 해시 기반 중복 체크 (파싱 없이 빠르게)
                is_processed, doc_id = await check_document_fully_processed(
                    rag, file_path, file_hash_cache
                )
                if is_processed:
                    print(f"  이미 처리됨 (doc_id: {doc_id[:16]}...) --force로 재처리 가능")
                    results["skipped"].append(str(file_path))
                    continue

            try:
                print(f"  처리 중...")
                result = await rag.process_document_complete(
                    file_path=str(file_path),
                    output_dir=OUTPUT_DIR,
                )

                # 처리 완료 후 파일 해시 캐시 업데이트
                if result and result.get("doc_id"):
                    update_file_hash_cache(file_hash_cache, file_path, result["doc_id"])

                print(f"  완료")
                results["processed"].append(str(file_path))

            except Exception as e:
                print(f"  실패: {e}")
                results["failed"].append({"file": str(file_path), "error": str(e)})

    finally:
        # 파일 해시 캐시 저장
        save_file_hash_cache(file_hash_cache)

        try:
            await rag.finalize_storages()
        except Exception:
            pass

    return results


def print_summary(results: dict):
    """처리 결과 요약 출력"""
    print("\n" + "=" * 60)
    print("인덱싱 완료")
    print("=" * 60)

    print(f"\n처리됨: {len(results['processed'])}개")
    for f in results["processed"]:
        print(f"  - {Path(f).name}")

    if results["skipped"]:
        print(f"\n건너뜀: {len(results['skipped'])}개")
        for f in results["skipped"]:
            print(f"  - {Path(f).name}")

    if results["failed"]:
        print(f"\n실패: {len(results['failed'])}개")
        for item in results["failed"]:
            print(f"  - {Path(item['file']).name}: {item['error']}")


async def main():
    parser = argparse.ArgumentParser(
        description="RAG-Anything Document Ingest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=str,
        help="PDF 파일 또는 폴더 경로",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 처리된 문서도 재처리",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="상세 로그 출력",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        type=str,
        nargs="+",
        default=["backup"],
        help="제외할 폴더/파일 패턴 (기본: backup)",
    )
    parser.add_argument(
        "--no-exclude",
        action="store_true",
        help="제외 패턴 비활성화 (모든 파일 포함)",
    )

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"경로를 찾을 수 없습니다: {path}")
        sys.exit(1)

    print("=" * 60)
    print("RAG-Anything Document Ingest")
    print("=" * 60)
    print(f"\n경로: {path}")
    print(f"작업 디렉토리: {WORKING_DIR}")
    print(f"출력 디렉토리: {OUTPUT_DIR}")

    if args.force:
        print("모드: 강제 재처리")

    # 제외 패턴 설정
    exclude_patterns = [] if args.no_exclude else args.exclude
    if exclude_patterns:
        print(f"제외 패턴: {', '.join(exclude_patterns)}")

    files = collect_pdf_files(path, exclude_patterns=exclude_patterns)

    if not files:
        print("\n처리할 PDF 파일이 없습니다.")
        sys.exit(0)

    print(f"\n발견된 PDF: {len(files)}개")
    for f in files:
        print(f"  - {f.name}")

    confirm = input("\n인덱싱을 시작하시겠습니까? [Y/n]: ")
    if confirm.lower() == "n":
        print("취소되었습니다.")
        sys.exit(0)

    results = await ingest_documents(
        files=files,
        force=args.force,
        verbose=args.verbose,
    )

    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
