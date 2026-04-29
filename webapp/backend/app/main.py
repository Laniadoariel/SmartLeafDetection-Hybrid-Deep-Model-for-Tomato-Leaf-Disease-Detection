"""SmartLeafDetection — FastAPI backend application."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import Base, engine
from app.routes.auth_routes import router as auth_router
from app.routes.flight_routes import router as flight_router

# Create all tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SmartLeafDetection API",
    version="1.0.0",
    description="Drone-based tomato leaf disease detection system",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(flight_router)

# Serve processing output files (frames, annotated images, crops)
output_dir = Path("processing_output")
output_dir.mkdir(exist_ok=True)
app.mount("/files/output", StaticFiles(directory=str(output_dir)), name="output")

uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)
app.mount("/files/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "SmartLeafDetection"}
