# Multi-stage build for production-ready ML inference service
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Copy source code
COPY src/ ./src/

# Install the package
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 mlsys && \
    chown -R mlsys:mlsys /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=mlsys:mlsys src/ ./src/
COPY --chown=mlsys:mlsys pyproject.toml ./

# Create directories for artifacts and data
RUN mkdir -p /app/artifacts /app/data && \
    chown -R mlsys:mlsys /app

# Copy model artifact (if exists) - this can be overridden at runtime
COPY --chown=mlsys:mlsys artifacts/ ./artifacts/ 2>/dev/null || true

# Switch to non-root user
USER mlsys

# Expose port
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Run the FastAPI application
CMD ["uvicorn", "mlsys.inference.service:app", "--host", "0.0.0.0", "--port", "8000"]
