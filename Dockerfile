FROM python:3.11-slim

WORKDIR /app

# git: clone hermes-agent during build (vendor/ is gitignored, not in COPY)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hermes is optional fallback for chat; openai in requirements.txt is the primary LLM path.
# vendor/hermes-agent is not in git — install_agent.sh shallow-clones it when missing.
ARG INSTALL_HERMES=1
RUN if [ "$INSTALL_HERMES" = "1" ]; then bash scripts/install_agent.sh; fi

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/exports /app/data /app/img \
    && chown -R appuser:appuser /app

USER appuser

ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8080\")}/health', timeout=4)"

CMD ["bash", "scripts/start_web.sh"]
