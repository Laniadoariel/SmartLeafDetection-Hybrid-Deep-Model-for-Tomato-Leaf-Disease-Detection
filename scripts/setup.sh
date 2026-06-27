#!/usr/bin/env bash
# Cross-platform setup for macOS / Linux.
# Creates a Python virtual environment, installs Python deps, and installs the
# frontend node modules. Run from anywhere; paths are resolved relative to repo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"

echo "==> Repo: $REPO_ROOT"
echo "==> Creating virtual environment (venv/) with $PYTHON"
"$PYTHON" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing core pipeline + training deps (requirements.txt)"
pip install -r requirements.txt

echo "==> Installing web backend deps (webapp/backend/requirements.txt)"
pip install -r webapp/backend/requirements.txt

if command -v npm >/dev/null 2>&1; then
  echo "==> Installing frontend deps (npm install)"
  (cd webapp/frontend && npm install)
else
  echo "!! npm not found — skipping frontend install. Install Node.js 18+ then run:"
  echo "     cd webapp/frontend && npm install"
fi

echo "==> Done. Activate the env with:  source venv/bin/activate"
