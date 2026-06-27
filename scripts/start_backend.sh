#!/usr/bin/env bash
# Start the FastAPI backend on macOS / Linux.
# Usage:  ./scripts/start_backend.sh [PORT]
#   LEAF_CONF=0.3 ./scripts/start_backend.sh 8000
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${1:-8000}"

# Activate venv if present (skip if already inside one, e.g. conda).
if [[ -z "${VIRTUAL_ENV:-}" && -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# Make the project root importable so `smart_leaf_detection` resolves.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Leaf-detector confidence (see README conf sweep); override by exporting first.
export LEAF_CONF="${LEAF_CONF:-0.3}"

cd webapp/backend
echo "==> Backend on http://localhost:${PORT}  (LEAF_CONF=${LEAF_CONF})"
exec python -m uvicorn app.main:app --port "$PORT" --reload
