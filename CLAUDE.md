# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Documentation

Comprehensive documentation is available in the `docs/` directory:

```
docs/
├── INDEX.md                 # Documentation index (start here)
├── BACKLOGS.md              # Project backlogs
├── guides/                  # Operation guides
│   ├── POSTGRES_RAG_GUIDE.md
│   ├── POSTGRES_MIGRATION.md
│   └── PROJECT_SETUP.md
├── reference/               # Technical reference
│   ├── API_REFERENCE.md
│   └── ARCHITECTURE.md
├── handoff/                 # Handoff documents
│   ├── HANDOFF.md
│   ├── MIGRATION_PLAN.md
│   └── MIGRATION_STATUS_20250108.md
└── reports/                 # Reports and benchmarks
    └── LATENCY_BY_TYPE_260108.md
```

Key documents:
- [docs/INDEX.md](docs/INDEX.md) - Start here for documentation overview
- [docs/reference/API_REFERENCE.md](docs/reference/API_REFERENCE.md) - HTTP and Python API
- [docs/reference/ARCHITECTURE.md](docs/reference/ARCHITECTURE.md) - System architecture
- [MIGRATION_PLAN.md](docs/handoff/MIGRATION_PLAN.md) - Containerization and deployment

## Build and Development Commands

```bash
# Install dependencies (use uv for this project)
uv sync
uv sync --all-extras                 # With all optional features

# Run examples
uv run python examples/raganything_example.py path/to/document.pdf --api-key YOUR_API_KEY
uv run python examples/modalprocessors_example.py --api-key YOUR_API_KEY

# Development tools (configured in pyproject.toml)
uv run pytest                        # Run tests
uv run black raganything/            # Format code
uv run isort raganything/            # Sort imports
uv run flake8 raganything/           # Lint
uv run mypy raganything/             # Type check

# Check parser installation
python -c "from raganything import RAGAnything; rag = RAGAnything(); print(rag.check_parser_installation())"
```

## Architecture Overview

RAG-Anything is a multimodal RAG framework built on top of LightRAG. It processes documents containing mixed content (text, images, tables, equations) and builds a multimodal knowledge graph.

### Core Pipeline Flow
```
Document Input -> Parser (MinerU/Docling) -> Content Separation ->
  -> Text: LightRAG insertion
  -> Multimodal: Modal Processors -> Knowledge Graph Integration
-> Query (text/multimodal/VLM-enhanced)
```

### Key Components

**raganything/raganything.py** - `RAGAnything` dataclass: Main entry point combining QueryMixin, ProcessorMixin, and BatchMixin. Manages LightRAG instance, modal processors, and context extraction.

**raganything/parser.py** - `MineruParser` and `DoclingParser` classes: Handle document parsing via CLI commands. MinerU supports PDF/images; Docling supports PDF/Office/HTML. Both convert Office docs via LibreOffice.

**raganything/modalprocessors.py** - Specialized processors:
- `BaseModalProcessor`: Base class with entity/chunk creation, JSON parsing, and belongs_to relationship handling
- `ImageModalProcessor`: Vision model integration for image analysis
- `TableModalProcessor`: Table structure interpretation
- `EquationModalProcessor`: LaTeX equation parsing
- `GenericModalProcessor`: Fallback for unknown types
- `ContextExtractor`: Extracts surrounding content for context-aware processing

**raganything/processor.py** - `ProcessorMixin`: Document processing logic including caching, batch multimodal processing, chunk template application, and doc_status management.

**raganything/query.py** - `QueryMixin`: Three query modes:
- `aquery()`: Text query via LightRAG (auto-enables VLM when vision_model_func available)
- `aquery_with_multimodal()`: Enhanced query with explicit multimodal content
- `aquery_vlm_enhanced()`: Replaces image paths in context with base64 for VLM processing

**raganything/config.py** - `RAGAnythingConfig` dataclass: All configuration with environment variable support. Key sections: directory, parsing, multimodal processing, context extraction, batch processing.

### Content List Format

The internal content representation uses dictionaries:
```python
{"type": "text", "text": "...", "page_idx": 0}
{"type": "image", "img_path": "/absolute/path.jpg", "image_caption": [...], "page_idx": 1}
{"type": "table", "table_body": "markdown", "table_caption": [...], "page_idx": 2}
{"type": "equation", "text": "LaTeX", "text_format": "...", "page_idx": 3}
```

### LightRAG Integration

RAGAnything wraps LightRAG and extends it with:
- Parse result caching via `parse_cache` KV storage
- Multimodal chunk processing with proper `doc_id` and `chunk_order_index`
- `belongs_to` relationships linking extracted entities to modal entities
- `multimodal_processed` flag in doc_status for tracking completion

### Key Patterns

**Async-first design**: All main methods are async (`aquery`, `process_document_complete`). Sync wrappers use `always_get_an_event_loop()`.

**Mixin composition**: `RAGAnything` combines functionality from QueryMixin, ProcessorMixin, and BatchMixin.

**Content-based doc_id**: Generated from content hash via `_generate_content_based_doc_id()` for consistent document identification.

**Chunk templates**: Modal content uses templates from `raganything/prompt.py` (image_chunk, table_chunk, equation_chunk, generic_chunk).

## Environment Variables

Key variables (see env.example for full list):
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`: API configuration
- `PARSER`: "mineru" or "docling"
- `PARSE_METHOD`: "auto", "ocr", or "txt"
- `WORKING_DIR`: RAG storage directory
- `OUTPUT_DIR`: Parser output directory

## External Dependencies

- **LibreOffice**: Required for Office document conversion (doc, docx, ppt, pptx, xls, xlsx)
- **MinerU**: Primary parser (`pip install mineru[core]`)
- **LightRAG**: Core RAG engine (`pip install lightrag-hku`)

## Project-Specific Scripts

This project extends upstream RAGAnything with PostgreSQL integration and operational scripts:

```bash
# Document ingestion (uses OpenAI for LLM)
uv run python scripts/ingest.py manuals/           # Index all PDFs
uv run python scripts/ingest.py manuals/ --force   # Force reprocess

# CLI Query (uses Gemini for LLM)
uv run python scripts/query.py "질문"              # Single query
uv run python scripts/query.py --interactive       # Interactive mode

# FastAPI Server (uses Gemini for LLM + VLM)
uv run python scripts/server.py                    # Start on port 9621

# Database cleanup
uv run python scripts/cleanup.py --dry-run         # Check current state
uv run python scripts/cleanup.py --all             # Full cleanup
```

## LLM Configuration

The project uses different LLM backends for different purposes:

| Script | LLM | Vision | Embedding |
|--------|-----|--------|-----------|
| `ingest.py` | OpenAI GPT-4.1-mini | OpenAI GPT-4.1 | text-embedding-3-small |
| `query.py` | Gemini 3 Flash | Gemini 3 Flash | text-embedding-3-small |
| `server.py` | Gemini 3 Flash | Gemini 3 Flash | text-embedding-3-small |

Required environment variables:
- `GEMINI_API_KEY`: For query and server
- `OPENAI_API_KEY`: For ingestion and embeddings

## PostgreSQL Backend

All data is stored in PostgreSQL using LightRAG storage backends:

```bash
# Start database container
cd docker && docker-compose up -d

# Verify extensions
docker exec postgres-rag psql -U pgvector -d ezis_rag -c "\dx"
# Expected: pgvector, age
```

Storage mapping:
- `PGKVStorage`: Documents, chunks, LLM cache
- `PGVectorStorage`: Embeddings (pgvector)
- `PGGraphStorage`: Knowledge graph (Apache AGE)
- `PGDocStatusStorage`: Processing status

## Upstream Modifications

Modified files in `upstream/raganything/`:

| File | Modification |
|------|--------------|
| `query.py` | VLM streaming support, `return_images=True` for image references |
| `prompt.py` | JSON formatting instructions (LaTeX escaping) |
| `processor.py` | `multimodal_processed` flag in metadata JSONB |

## Directory Structure Reference

```
RAG-Anything-1/
├── scripts/           # Operational scripts (ingest, query, server, cleanup)
├── upstream/          # RAGAnything source (modified)
│   └── raganything/
├── docker/            # PostgreSQL container config
├── demo-front/        # Web UI (chat interface)
├── docs/              # Project documentation
├── manuals/           # EZIS product manuals (PDF)
├── output/            # Parsed content (images, markdown)
└── rag_storage/       # LightRAG local cache (unused with PG backend)
```
