# System Architecture

## Overview

EZIS RAG는 LightRAG 기반의 멀티모달 RAG 시스템으로, PostgreSQL 백엔드를 활용하여 지식 그래프 기반 검색을 수행한다.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                    │
│  ┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐ │
│  │  demo-front/    │  │  scripts/query.py   │  │  External Applications  │ │
│  │  (Web UI)       │  │  (CLI)              │  │  (REST API clients)     │ │
│  └────────┬────────┘  └──────────┬──────────┘  └────────────┬────────────┘ │
└───────────┼──────────────────────┼──────────────────────────┼──────────────┘
            │                      │                          │
            │ HTTP/SSE             │ Direct                   │ HTTP
            ▼                      ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API Layer (scripts/server.py)                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Application                                                   │  │
│  │  ├── /health              → Health check                              │  │
│  │  ├── /storage/info        → Storage backend info                      │  │
│  │  ├── /query               → Synchronous query                         │  │
│  │  └── /query/stream        → NDJSON streaming query                    │  │
│  │                                                                        │  │
│  │  Features:                                                             │  │
│  │  - CORS middleware (allow all origins)                                │  │
│  │  - Static file serving (/static → output/)                            │  │
│  │  - Lifespan management (RAGAnything init/cleanup)                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAGAnything Core Layer                                │
│                    (upstream/raganything/)                                   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                    RAGAnything (Dataclass)                              ││
│  │          Inherits: QueryMixin + ProcessorMixin + BatchMixin             ││
│  │                                                                          ││
│  │  Core Components:                                                        ││
│  │  ├── lightrag: LightRAG instance                                        ││
│  │  ├── llm_model_func: async LLM callable                                 ││
│  │  ├── vision_model_func: async VLM callable                              ││
│  │  ├── embedding_func: EmbeddingFunc                                      ││
│  │  ├── config: RAGAnythingConfig                                          ││
│  │  └── modal_processors: Dict[str, ModalProcessor]                        ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │   QueryMixin     │  │  ProcessorMixin  │  │      BatchMixin          │  │
│  │                  │  │                  │  │                          │  │
│  │  - aquery()      │  │  - parse_doc()   │  │  - process_folder()      │  │
│  │  - aquery_multi  │  │  - insert_text() │  │  - batch_processing()    │  │
│  │    modal()       │  │  - process_      │  │  - concurrent file       │  │
│  │  - aquery_vlm_   │  │    multimodal()  │  │    handling              │  │
│  │    enhanced()    │  │  - caching       │  │                          │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Modal Processors                                   │  │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────────────┐│  │
│  │  │  Image    │ │  Table    │ │ Equation  │ │      Generic          ││  │
│  │  │ Processor │ │ Processor │ │ Processor │ │     Processor         ││  │
│  │  │           │ │           │ │           │ │                       ││  │
│  │  │ Vision    │ │ LLM       │ │ LLM       │ │ LLM                   ││  │
│  │  │ Model     │ │ Analysis  │ │ LaTeX     │ │ Fallback              ││  │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────────────────┘│  │
│  │                                                                      │  │
│  │  Common: ContextExtractor (page/chunk based context window)          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                       Parsers                                         │  │
│  │  ┌────────────────────────┐  ┌──────────────────────────────────────┐│  │
│  │  │      MineruParser      │  │         DoclingParser                ││  │
│  │  │                        │  │                                      ││  │
│  │  │  - PDF, Images         │  │  - PDF, Office (doc/ppt/xls)         ││  │
│  │  │  - CLI: magic-pdf      │  │  - HTML                              ││  │
│  │  │  - parse_method:       │  │  - Uses docling library              ││  │
│  │  │    auto/ocr/txt        │  │                                      ││  │
│  │  └────────────────────────┘  └──────────────────────────────────────┘│  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LightRAG Layer                                      │
│                     (lightrag-hku package)                                   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    LightRAG Core                                      │  │
│  │                                                                        │  │
│  │  Query Modes:                                                          │  │
│  │  ├── naive:  Vector similarity search only                            │  │
│  │  ├── local:  Entity-centric retrieval                                 │  │
│  │  ├── global: Community/relationship-based retrieval                   │  │
│  │  ├── hybrid: local + global combination                               │  │
│  │  └── mix:    hybrid + naive (default, recommended)                    │  │
│  │                                                                        │  │
│  │  Knowledge Graph Operations:                                           │  │
│  │  ├── Entity extraction (LLM-based)                                    │  │
│  │  ├── Relationship extraction                                          │  │
│  │  ├── Community detection                                              │  │
│  │  └── Graph traversal for context retrieval                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌──────────────┐ │
│  │  KV Storage    │ │ Vector Storage │ │ Graph Storage  │ │  DocStatus   │ │
│  │  PGKVStorage   │ │ PGVectorStorage│ │ PGGraphStorage │ │  Storage     │ │
│  └───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └──────┬───────┘ │
└──────────┼──────────────────┼──────────────────┼─────────────────┼──────────┘
           │                  │                  │                 │
           └──────────────────┼──────────────────┼─────────────────┘
                              │                  │
                              ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PostgreSQL 16 Layer                                   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Core PostgreSQL                                    │  │
│  │  ├── Tables: lightrag_doc_*, lightrag_vdb_*, lightrag_full_*         │  │
│  │  └── Standard SQL operations, JSONB support                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────┐  ┌───────────────────────────────────────┐  │
│  │       pgvector 0.8.1      │  │         Apache AGE 1.5.0              │  │
│  │                           │  │                                       │  │
│  │  - Vector embedding       │  │  - Graph database                     │  │
│  │    storage                │  │  - Cypher query language              │  │
│  │  - Cosine similarity      │  │  - Entity-Relationship graphs         │  │
│  │  - HNSW/IVFFlat indexes   │  │  - Community detection                │  │
│  │  - Max 2000 dims (HNSW)   │  │  - Graph traversal                    │  │
│  └───────────────────────────┘  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Document Ingestion Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PDF       │────▶│   Parser    │────▶│  Content    │────▶│   Text      │
│   Input     │     │   (MinerU)  │     │   List      │     │   Chunks    │
└─────────────┘     └─────────────┘     └──────┬──────┘     └──────┬──────┘
                                               │                    │
                                               │                    ▼
                                               │           ┌─────────────┐
                                               │           │  LightRAG   │
                                               │           │  Insert     │
                                               │           │  (Text)     │
                                               │           └──────┬──────┘
                                               │                  │
                                               ▼                  ▼
                                        ┌─────────────┐   ┌─────────────┐
                                        │  Multimodal │   │  Entity &   │
                                        │  Content    │   │  Relation   │
                                        │  (Images,   │   │  Extraction │
                                        │  Tables)    │   │  (LLM)      │
                                        └──────┬──────┘   └──────┬──────┘
                                               │                 │
                                               ▼                 ▼
                                        ┌─────────────┐   ┌─────────────┐
                                        │   Modal     │   │  Knowledge  │
                                        │  Processors │   │    Graph    │
                                        │  (VLM/LLM)  │   │   Build     │
                                        └──────┬──────┘   └──────┬──────┘
                                               │                 │
                                               ▼                 ▼
                                        ┌─────────────────────────────┐
                                        │        PostgreSQL           │
                                        │  (KV + Vector + Graph)      │
                                        └─────────────────────────────┘
```

### Query Flow (VLM Enhanced)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   User      │────▶│  QueryMixin │────▶│  LightRAG   │
│   Query     │     │  aquery()   │     │  Retrieval  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Raw        │
                                        │  Context    │
                                        │  (with      │
                                        │  img paths) │
                                        └──────┬──────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
             ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
             │  Image Path │           │   Text      │           │   Build     │
             │  Detection  │           │   Context   │           │   VLM       │
             │  (Regex)    │           │             │           │   Messages  │
             └──────┬──────┘           └─────────────┘           └──────┬──────┘
                    │                                                   │
                    ▼                                                   │
             ┌─────────────┐                                           │
             │   Base64    │                                           │
             │   Encode    │───────────────────────────────────────────┘
             │   Images    │
             └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │     VLM     │
                                        │   (Gemini)  │
                                        │   Response  │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Parse     │
                                        │[REFERENCED_ │
                                        │ IMAGES: ]   │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Return     │
                                        │  {response, │
                                        │   images[]} │
                                        └─────────────┘
```

---

## Storage Architecture

### PostgreSQL Tables

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KV Storage                                         │
│                                                                              │
│  lightrag_doc_full          lightrag_doc_chunks       lightrag_llm_cache    │
│  ┌────────────────┐        ┌────────────────┐        ┌────────────────┐     │
│  │ id (PK)        │        │ id (PK)        │        │ id (PK)        │     │
│  │ workspace      │        │ workspace      │        │ workspace      │     │
│  │ content (TEXT) │        │ content (TEXT) │        │ original_prompt│     │
│  │ metadata (JSONB│        │ doc_id (FK)    │        │ return (TEXT)  │     │
│  └────────────────┘        │ chunk_order_idx│        │ model          │     │
│                            │ metadata (JSONB│        │ metadata (JSONB│     │
│                            └────────────────┘        └────────────────┘     │
│                                                                              │
│  lightrag_doc_status                                                         │
│  ┌────────────────┐                                                         │
│  │ id (PK)        │  metadata JSONB includes:                               │
│  │ workspace      │  - multimodal_processed: bool                           │
│  │ file_path      │  - content_summary: str                                 │
│  │ status         │  - processing_time: float                               │
│  │ metadata (JSONB│                                                         │
│  └────────────────┘                                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Vector Storage (pgvector)                            │
│                                                                              │
│  lightrag_vdb_chunks         lightrag_vdb_entity      lightrag_vdb_relation │
│  ┌────────────────┐        ┌────────────────┐        ┌────────────────┐     │
│  │ id (PK)        │        │ id (PK)        │        │ id (PK)        │     │
│  │ workspace      │        │ workspace      │        │ workspace      │     │
│  │ embedding      │        │ embedding      │        │ embedding      │     │
│  │ (vector(1536)) │        │ (vector(1536)) │        │ (vector(1536)) │     │
│  │ content_key    │        │ entity_name    │        │ relation_name  │     │
│  └────────────────┘        └────────────────┘        └────────────────┘     │
│                                                                              │
│  Index: HNSW (m=16, ef_construction=200)                                    │
│  Similarity: cosine                                                          │
│  Note: HNSW limited to 2000 dimensions                                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Graph Storage (Apache AGE)                           │
│                                                                              │
│  Graph: chunk_entity_relation                                               │
│                                                                              │
│  Vertices:                          Edges:                                  │
│  ┌────────────────┐                ┌────────────────┐                       │
│  │ :Entity        │                │ :RELATED_TO    │                       │
│  │ - name         │◀───────────────│ - weight       │                       │
│  │ - type         │                │ - description  │                       │
│  │ - description  │                │ - source_id    │                       │
│  │ - source_id    │                └────────────────┘                       │
│  └────────────────┘                                                         │
│                                                                              │
│  Mapping Tables:                                                            │
│  - lightrag_entity_chunks: Entity → Chunk mappings                          │
│  - lightrag_relation_chunks: Relation → Chunk mappings                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### RAGAnythingConfig

```python
@dataclass
class RAGAnythingConfig:
    # Directory
    working_dir: str = "./rag_storage"
    parser_output_dir: str = "./output"

    # Parser
    parser: str = "mineru"          # mineru | docling
    parse_method: str = "auto"      # auto | ocr | txt

    # Multimodal Processing
    enable_image_processing: bool = True
    enable_table_processing: bool = True
    enable_equation_processing: bool = True

    # Context Extraction
    context_window: int = 1
    context_mode: str = "page"      # page | chunk
    max_context_tokens: int = 2000
    include_headers: bool = True
    include_captions: bool = True

    # Batch Processing
    max_concurrent_files: int = 1
    supported_file_extensions: List[str] = [".pdf", ".jpg", ...]
    recursive_folder_processing: bool = True
```

### Modal Processor Base Class

```python
class BaseModalProcessor:
    def __init__(
        self,
        lightrag: LightRAG,
        modal_caption_func: Callable,
        context_extractor: ContextExtractor = None
    ):
        self.lightrag = lightrag
        self.modal_caption_func = modal_caption_func
        self.context_extractor = context_extractor

    async def process(
        self,
        content_item: Dict[str, Any],
        doc_id: str,
        chunk_order_index: int
    ) -> Dict[str, Any]:
        """
        Process a single modal content item.
        Returns processed chunk with entities and relationships.
        """
        pass

    def _create_entity(self, name: str, entity_type: str, description: str) -> Dict:
        """Create entity dict for knowledge graph insertion."""
        pass

    def _create_belongs_to_relationship(self, entity_name: str, parent_doc_id: str):
        """Create belongs_to relationship to link entity to document."""
        pass
```

### ContextExtractor

```python
class ContextExtractor:
    def __init__(self, config: ContextConfig, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer

    def extract_context(
        self,
        content_source: Any,
        current_item_info: Dict[str, Any],
        content_format: str = "auto"
    ) -> str:
        """
        Extract surrounding context for current item.

        Modes:
        - page: Extract from surrounding pages (page_idx based)
        - chunk: Extract from surrounding content items (index based)

        Returns truncated context text within max_context_tokens.
        """
        pass
```

---

## LLM Integration

### Model Function Signatures

```python
# LLM Function (text generation)
async def llm_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[Dict] = [],
    **kwargs
) -> Union[str, AsyncIterator[str]]:
    """
    Args:
        prompt: User prompt
        system_prompt: System instructions
        history_messages: Previous conversation turns
        **kwargs: Model-specific parameters

    Returns:
        str: Generated text
        AsyncIterator[str]: If stream=True
    """
    pass

# Vision Model Function (multimodal)
async def vision_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[Dict] = [],
    image_data: str = None,         # Base64 encoded image
    messages: List[Dict] = None,    # OpenAI-style multimodal messages
    **kwargs
) -> Union[str, AsyncIterator[str]]:
    """
    Args:
        image_data: Single base64 image
        messages: List of multimodal messages for batch images
    """
    pass

# Embedding Function
embedding_func = EmbeddingFunc(
    embedding_dim=1536,
    max_token_size=8192,
    func=lambda texts: embed_api_call(texts)  # Returns List[List[float]]
)
```

### Current Model Configuration

| Component | Development | Production |
|-----------|-------------|------------|
| LLM (Ingest) | OpenAI GPT-4.1-mini | OpenAI GPT-4.1-mini |
| LLM (Query/Server) | Gemini 3 Flash | Gemini 3 Flash |
| Vision | Gemini 3 Flash | Gemini 3 Flash |
| Embedding | OpenAI text-embedding-3-small | OpenAI text-embedding-3-small |

---

## Deployment Architecture

### Local Development

```
┌─────────────────────────────────────────┐
│           Developer Machine              │
│                                          │
│  ┌──────────────────┐ ┌──────────────┐  │
│  │  Python Scripts  │ │ demo-front/  │  │
│  │  (server.py,     │ │ (Static)     │  │
│  │   query.py)      │ │              │  │
│  └────────┬─────────┘ └──────────────┘  │
│           │                              │
│           │ localhost:5432               │
│           ▼                              │
│  ┌──────────────────────────────────┐   │
│  │      Docker Container            │   │
│  │      postgres-rag                │   │
│  │      (pgvector + AGE)            │   │
│  │                                  │   │
│  │      Port: 5432                  │   │
│  │      Volume: postgres_data       │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Production (Containerized)

```
┌─────────────────────────────────────────────────────────────┐
│                    192.168.0.47 Server                       │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  Docker Network: ezis-network           │ │
│  │                                                          │ │
│  │  ┌──────────────────────┐  ┌──────────────────────────┐│ │
│  │  │   ezis-rag-app       │  │   ezis-postgres-rag      ││ │
│  │  │   (FastAPI)          │  │   (pgvector + AGE)       ││ │
│  │  │                      │  │                          ││ │
│  │  │   Port: 8000         │  │   Port: 5432             ││ │
│  │  │   Exposed: 38000     │  │   Exposed: 35432         ││ │
│  │  │                      │  │                          ││ │
│  │  │   Volumes:           │  │   Volume:                ││ │
│  │  │   - rag_storage      │  │   - postgres_data        ││ │
│  │  │   - output_data      │  │                          ││ │
│  │  └──────────┬───────────┘  └─────────────────────────┘ │ │
│  │             │                        ▲                  │ │
│  │             │   postgres-rag:5432    │                  │ │
│  │             └────────────────────────┘                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  External Access:                                            │
│  - API: http://192.168.0.47:38000                           │
│  - DB:  postgresql://...@192.168.0.47:35432/ezis_rag        │
└─────────────────────────────────────────────────────────────┘
```

---

## Security Considerations

### Current Implementation

| Aspect | Status | Notes |
|--------|--------|-------|
| CORS | Open (`*`) | Tighten for production |
| DB Password | Default (`pgvector`) | Change for production |
| API Auth | None | Add JWT/API key for production |
| TLS | None | Add reverse proxy with TLS |
| Secrets | `.env` file | Use secrets manager |

### Recommended Production Setup

```
┌─────────────────────────────────────────────────────────────┐
│                     Reverse Proxy (nginx)                    │
│                     TLS termination                          │
│                     Rate limiting                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     API Gateway                              │
│                     JWT validation                           │
│                     CORS whitelist                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     ezis-rag-app                             │
│                     Internal network only                    │
└─────────────────────────────────────────────────────────────┘
```

---

*Last Updated: 2026-01-13*
