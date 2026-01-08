"""
RAG-Anything Demo with PostgreSQL + OpenAI

This example demonstrates how to use RAG-Anything with:
- OpenAI (LLM + Vision + Embeddings)
- PostgreSQL-backed storages for:
  - Vector storage (pgvector)
  - Graph storage (Apache AGE)
  - KV storage
  - Document status storage

Prerequisites:
1. PostgreSQL container running: cd docker && docker-compose up -d
2. Set OPENAI_API_KEY environment variable or create .env file

Usage:
    python scripts/raganything_openai_postgres_demo.py
"""

import os
import asyncio

from dotenv import load_dotenv

load_dotenv()

# PostgreSQL configuration (match docker/docker-compose.yml)
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
WORKING_DIR = "./rag_storage"
TEST_PDF = "manuals/test.pdf"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")


# Model configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4.1")


# LLM function (OpenAI)
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


# Vision model function (OpenAI)
async def vision_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    image_data=None,
    messages=None,
    **kwargs,
) -> str:
    if messages:
        # Multimodal VLM enhanced query format
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
        # Traditional single image format
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
        # Pure text format
        return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)


# Embedding configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# Embedding function (OpenAI)
embedding_func = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM,
    max_token_size=8192,
    func=lambda texts: openai_embed(
        texts,
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    ),
)


async def main():
    rag = None
    try:
        print("=" * 60)
        print("RAG-Anything Demo: OpenAI + PostgreSQL")
        print("=" * 60)

        # Check test file
        if not os.path.exists(TEST_PDF):
            raise FileNotFoundError(f"Test file not found: {TEST_PDF}")

        print(f"\nTest PDF: {TEST_PDF}")
        print(f"Working directory: {WORKING_DIR}")

        # Create working directory
        os.makedirs(WORKING_DIR, exist_ok=True)

        # Initialize RAGAnything with PostgreSQL storages
        print("\nInitializing RAG-Anything with PostgreSQL backend...")
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

        # Process document
        print(f"\nProcessing document: {TEST_PDF}")
        print("This may take a few minutes...")
        await rag.process_document_complete(
            file_path=TEST_PDF,
            output_dir="./output",
        )
        print("Document processed successfully!")

        # Run queries
        print("\n" + "=" * 60)
        print("Running sample queries")
        print("=" * 60)

        query = "이 문서의 주요 내용은 무엇인가요?"

        for mode in ["naive", "local", "global", "hybrid", "mix"]:
            print(f"\n[{mode.upper()} MODE]")
            try:
                result = await rag.aquery(query, mode=mode)
                # Truncate long results
                display = result[:500] + "..." if len(result) > 500 else result
                print(display)
            except Exception as e:
                print(f"Error in {mode} mode: {e}")

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if rag is not None:
            try:
                await rag.finalize()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
