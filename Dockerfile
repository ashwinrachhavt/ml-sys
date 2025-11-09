# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --upgrade pip

COPY src ./src
COPY config ./config
COPY dataset ./dataset
COPY scripts ./scripts
COPY README.md ./

RUN pip install --no-cache-dir .

RUN adduser --disabled-login --gecos "" mlsys && chown -R mlsys:mlsys /app
USER mlsys

ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
