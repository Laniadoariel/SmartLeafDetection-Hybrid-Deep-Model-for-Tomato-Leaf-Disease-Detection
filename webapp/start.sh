#!/bin/bash
# SmartLeafDetection — Start both backend and frontend
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🌿 SmartLeafDetection Web Application"
echo "======================================"

# --- Backend setup (use existing project venv) ---
echo ""
echo "📦 Setting up backend..."
cd "$SCRIPT_DIR/backend"

source "$PROJECT_ROOT/venv/bin/activate"

# Install only the web-specific packages into the existing venv
pip install -q fastapi uvicorn python-multipart sqlalchemy python-jose passlib aiofiles 2>/dev/null || true

echo "✅ Backend ready"

# Start backend server
echo "🚀 Starting backend on http://localhost:8000"
PYTHONPATH="$SCRIPT_DIR/backend:$PROJECT_ROOT" \
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# --- Frontend setup ---
echo ""
echo "📦 Setting up frontend..."
cd "$SCRIPT_DIR/frontend"

if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

echo "🚀 Starting frontend on http://localhost:3000"
npm run dev &
FRONTEND_PID=$!

# Wait a moment then open browser
sleep 4
echo ""
echo "======================================"
echo "🌿 SmartLeafDetection is running!"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API docs: http://localhost:8000/docs"
echo "======================================"

# Open browser
if command -v open &> /dev/null; then
    open http://localhost:3000
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:3000
fi

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
