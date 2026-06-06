#!/usr/bin/env bash
# Install Nous Research Hermes Agent for market exposure analysis.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor/hermes-agent"

if [[ ! -f "$VENDOR/pyproject.toml" ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "Hermes Agent source not found at $VENDOR and git is unavailable." >&2
    echo "Vendor the repo under vendor/hermes-agent or install git." >&2
    exit 1
  fi
  rm -rf "$VENDOR"
  git clone --depth 1 https://github.com/NousResearch/hermes-agent.git "$VENDOR"
fi

pip install -e "$VENDOR"
echo "Hermes Agent installed. Set OPENAI_API_KEY or OPENROUTER_API_KEY for LLM analysis."
echo "Optional: GEX_HERMES_PROVIDER=openai|openrouter, GEX_AGENT_MODEL=<model>"
