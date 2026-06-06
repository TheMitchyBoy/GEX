#!/usr/bin/env bash
# Install the gex-llm-patterns market exposure agent from GitHub.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor/gex-llm-patterns"

if [[ ! -d "$VENDOR/.git" ]]; then
  git clone --depth 1 https://github.com/iAmGiG/gex-llm-patterns.git "$VENDOR"
fi

pip install -r "$ROOT/requirements-agent.txt"
echo "Agent installed. Set OPENAI_API_KEY for LLM-enhanced analysis."
