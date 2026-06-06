#!/usr/bin/env bash
# Install Nous Research Hermes Agent for market exposure analysis.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor/hermes-agent"

if [[ ! -d "$VENDOR/.git" ]]; then
  git clone --depth 1 https://github.com/NousResearch/hermes-agent.git "$VENDOR"
fi

pip install -e "$VENDOR"
echo "Hermes Agent installed. Set OPENAI_API_KEY or OPENROUTER_API_KEY for LLM analysis."
echo "Optional: GEX_HERMES_PROVIDER=openai|openrouter, GEX_AGENT_MODEL=<model>"
