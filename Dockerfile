FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"
COPY templates ./templates
COPY static ./static
COPY data ./data
ENV PATH="/app/.venv/bin:$PATH"
CMD ["sh", "-c", "uvicorn nextbook.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
