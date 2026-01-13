# EZIS RAG Application - FastAPI Server
# Multi-stage build for optimized image size

FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies (including OpenCV runtime deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first for better caching
COPY requirements.txt .
COPY upstream/ ./upstream/

# Install dependencies (ensure mineru CLI is installed)
RUN uv pip install --system --no-cache -r requirements.txt && \
    uv pip install --system --no-cache "mineru[core]" && \
    which mineru && mineru --version

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only (including OpenCV deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY upstream/ ./upstream/
COPY scripts/ ./scripts/

# Create directories for runtime data
RUN mkdir -p /app/rag_storage /app/output

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/upstream
ENV WORKING_DIR=/app/rag_storage
ENV OUTPUT_DIR=/app/output

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
CMD ["python", "scripts/server.py", "--host", "0.0.0.0", "--port", "8000"]
