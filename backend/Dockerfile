# syntax=docker/dockerfile:1
FROM python:3.12-slim

# postgresql-client -> pg_isready, used by scripts/entrypoint.sh to wait for the DB
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.20 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml* uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-progress -vv

COPY . .
RUN uv sync

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src/ai_customer_assistant"

RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["scripts/entrypoint.sh"]

# 1. Prevent hard-link failures across cache boundaries and turn on bytecode compilation
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# 2. Add an explicit cache ID to prevent re-downloads on failure
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-progress

COPY . .

# 3. Mount the same cache for the final project sync so it doesn't re-download CUDA wheels
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv sync