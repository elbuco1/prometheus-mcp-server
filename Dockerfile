FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml ./
COPY uv.lock ./

COPY src ./src/

RUN uv venv && \
    uv sync --frozen --no-dev && \
    uv pip install -e . --no-deps && \
    uv pip install --upgrade pip setuptools

FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        procps \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get autoremove -y

RUN groupadd -r -g 1000 app && \
    useradd -r -g app -u 1000 -d /app -s /bin/false app && \
    chown -R app:app /app && \
    chmod 755 /app && \
    chmod -R go-w /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --chown=app:app pyproject.toml /app/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    PYTHONFAULTHANDLER=1 \
    METRICS_MCP_BIND_HOST=0.0.0.0 \
    METRICS_MCP_BIND_PORT=8080

USER app

EXPOSE 8080

CMD ["/app/.venv/bin/metrics-mcp-server"]