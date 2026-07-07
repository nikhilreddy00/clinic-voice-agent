# Clinic voice agent (Pipecat). Defaults to MODE=telephony (LiveKit SIP) — see docker-compose.yml.
# All config (API keys, LiveKit creds, scheduling API URL) comes from the environment at run time;
# nothing secret is baked into the image (see .dockerignore — .env is excluded).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# pyaudio (pulled in by pipecat's [local] extra, imported at module load even in telephony mode)
# needs PortAudio headers + a C toolchain to build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first (cached until the lockfile changes), then the package.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Runtime defaults. MODE=telephony so the container joins the LiveKit SIP room by default;
# SCHEDULING_API_BASE_URL points at the compose service name, not localhost. Secrets are injected
# via env_file at run time, never here.
ENV MODE=telephony \
    CLINIC_LOG_DIR=/logs \
    SCHEDULING_API_BASE_URL=http://scheduling_api:8000

CMD ["uv", "run", "python", "-m", "clinic_agent.pipeline"]
