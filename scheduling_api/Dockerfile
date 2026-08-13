# Mock scheduling API (FastAPI + SQLite). Small, stateless-ish (SQLite is re-seeded on start).
FROM python:3.12-slim

# uv for fast, locked installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Install dependencies first (cached until the lockfile changes), then the app itself.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
RUN uv sync --frozen --no-dev

# calls.jsonl is written by the agent and read here; the dashboard/ dir is mounted in.
# Both paths are provided by docker-compose as a shared volume + bind mount.
ENV CLINIC_LOG_DIR=/logs \
    CLINIC_DASHBOARD_DIR=/app/dashboard \
    # CLINIC_DATABASE_URL is injected at run time (it carries a password) — never baked in.

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
