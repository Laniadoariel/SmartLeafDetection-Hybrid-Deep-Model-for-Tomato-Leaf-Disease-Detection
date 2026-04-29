# SmartLeafDetection — Web Application

Drone-based tomato leaf disease detection system with real ML pipeline integration.

## Quick Start

```bash
cd webapp
./start.sh
```

This will:
1. Create a Python venv and install backend dependencies
2. Install frontend npm packages
3. Start the FastAPI backend on port 8000
4. Start the React frontend on port 3000
5. Open the browser automatically

## Manual Start

### Backend
```bash
cd webapp/backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install ultralytics opencv-python torch torchvision numpy pyyaml
PYTHONPATH=".:$(pwd)/../.." uvicorn app.main:app --port 8000 --reload
```

### Frontend
```bash
cd webapp/frontend
npm install
npm run dev
```

## Architecture

```
webapp/
  backend/
    app/
      main.py          — FastAPI application entry point
      database.py      — SQLAlchemy setup (SQLite/MySQL)
      models.py        — ORM models (User, Flight, PlantResult, LeafResult, FrameRecord)
      schemas.py       — Pydantic request/response schemas
      auth.py          — JWT authentication + password hashing
      deps.py          — Dependency injection (current user)
      worker.py        — Background pipeline worker (real ML inference)
      routes/
        auth_routes.py   — Login / Signup
        flight_routes.py — Upload / Start / History / Details
  frontend/
    src/
      App.tsx          — Router with protected routes
      api.ts           — Axios client + TypeScript interfaces
      pages/
        Login.tsx      — Login page
        Signup.tsx     — Registration page
        Dashboard.tsx  — Main 3-tab dashboard
      components/
        UploadTab.tsx       — Video upload + processing timeline
        InvestigationTab.tsx — Frame gallery + detection overlays
        ResultsTab.tsx      — Disease summary + plant cards
        HistoryTab.tsx      — Previous flight analyses
```

## Real Pipeline Integration

The backend worker (`app/worker.py`) uses the actual trained YOLO models
from the project to process uploaded videos:

1. Video decode + frame extraction (OpenCV)
2. Disease detection per frame (YOLOv11 trained model)
3. Annotated frame generation with bounding boxes
4. Plant-level result aggregation
5. Database persistence (SQLite/MySQL)
6. Real-time progress updates via polling

## API Endpoints

- `POST /api/auth/signup` — Create account
- `POST /api/auth/login` — Get JWT token
- `POST /api/flights/upload` — Upload drone video
- `POST /api/flights/{id}/start` — Start analysis
- `GET /api/flights/history` — List previous flights
- `GET /api/flights/{id}` — Full flight details with results
- `GET /api/health` — Health check
