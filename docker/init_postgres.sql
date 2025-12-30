-- PostgreSQL initialization script for RAGAnything
-- Extensions: pgvector (vector similarity) + Apache AGE (graph database)

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable Apache AGE extension for graph database
CREATE EXTENSION IF NOT EXISTS age;

-- Load AGE into search path
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Create default graph for LightRAG (will be created per workspace)
-- Note: LightRAG will create workspace-specific graphs automatically

-- Grant necessary permissions
GRANT USAGE ON SCHEMA ag_catalog TO pgvector;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ag_catalog TO pgvector;

-- Verify extensions
DO $$
BEGIN
    RAISE NOTICE 'PostgreSQL RAG Database initialized successfully';
    RAISE NOTICE 'pgvector extension: enabled';
    RAISE NOTICE 'Apache AGE extension: enabled';
END $$;
