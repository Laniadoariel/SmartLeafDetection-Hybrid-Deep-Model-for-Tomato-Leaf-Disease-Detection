#!/usr/bin/env bash
# Start the React/Vite frontend dev server on macOS / Linux.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/webapp/frontend"

if [[ ! -d node_modules ]]; then
  echo "==> node_modules missing — running npm install"
  npm install
fi

echo "==> Frontend dev server starting (Vite)"
exec npm run dev
