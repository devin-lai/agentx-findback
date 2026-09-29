# Baseline application image. Deploy CUDA/model services in a validated GPU environment.
FROM node:22-bookworm-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim-trixie AS app
COPY --from=ghcr.io/astral-sh/uv:0.11.33 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-cache
COPY --from=web /web/dist ./web/dist
RUN useradd --uid 10001 --create-home agentx && mkdir /data && chown agentx:agentx /data
ENV AGENTX_DATA_DIR=/data AGENTX_WEB_DIST=/app/web/dist PYTHONUNBUFFERED=1
USER agentx
EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000/api/health', timeout=3)"]
CMD ["/app/.venv/bin/agentx", "serve", "--host", "0.0.0.0", "--port", "9000"]
