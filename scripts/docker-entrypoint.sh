#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/data/exports /app/img

# Railway/bind mounts often arrive root-owned; fix ownership when we still have root.
if [ "$(id -u)" = "0" ]; then
  chown -R appuser:appuser /app/data /app/img 2>/dev/null || true
  exec setpriv --reuid=appuser --regid=appuser --init-groups -- "$@"
fi

exec "$@"
