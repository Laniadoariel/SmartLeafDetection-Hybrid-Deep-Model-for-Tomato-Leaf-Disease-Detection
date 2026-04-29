"""Flight management routes — upload, status, history, details."""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Flight, FrameRecord, LeafResult, PlantResult, User
from app.schemas import (
    FlightDetailResponse,
    FlightSummary,
    FrameResponse,
    LeafResultResponse,
    PlantResultResponse,
)

router = APIRouter(prefix="/api/flights", tags=["flights"])

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "processing_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=FlightSummary)
def upload_video(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(400, f"Unsupported format: {ext}")

    flight = Flight(user_id=user.id, video_filename=file.filename or "video.mp4", video_path="")
    db.add(flight)
    db.flush()

    # Save uploaded file
    flight_dir = UPLOAD_DIR / flight.id
    flight_dir.mkdir(parents=True, exist_ok=True)
    video_path = flight_dir / (file.filename or "video.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    flight.video_path = str(video_path)
    db.commit()
    db.refresh(flight)
    return _to_summary(flight)


@router.post("/{flight_id}/start")
def start_analysis(
    flight_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    flight = db.query(Flight).filter(Flight.id == flight_id, Flight.user_id == user.id).first()
    if not flight:
        raise HTTPException(404, "Flight not found")
    if flight.status not in ("uploaded", "failed"):
        raise HTTPException(400, f"Cannot start analysis: status is {flight.status}")

    flight.status = "processing"
    flight.current_stage = "Initializing"
    flight.progress = 0.0
    flight.error_message = None
    db.commit()

    # Launch processing in background thread
    from app.worker import run_pipeline_async
    thread = threading.Thread(target=run_pipeline_async, args=(flight.id,), daemon=True)
    thread.start()

    return {"status": "started", "flight_id": flight_id}


@router.get("/history", response_model=list[FlightSummary])
def flight_history(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    flights = (
        db.query(Flight)
        .filter(Flight.user_id == user.id)
        .order_by(Flight.created_at.desc())
        .all()
    )
    return [_to_summary(f) for f in flights]


@router.get("/{flight_id}", response_model=FlightDetailResponse)
def flight_detail(
    flight_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    flight = db.query(Flight).filter(Flight.id == flight_id, Flight.user_id == user.id).first()
    if not flight:
        raise HTTPException(404, "Flight not found")

    plants = []
    for pr in flight.plant_results:
        leaves = [
            LeafResultResponse(
                leaf_id=lr.leaf_id, label=lr.label, confidence=lr.confidence,
                bbox=[lr.bbox_x1, lr.bbox_y1, lr.bbox_x2, lr.bbox_y2],
                crop_path=lr.crop_path,
            )
            for lr in pr.leaf_results
        ]
        labels = [l.strip() for l in (pr.disease_labels or "").split(",") if l.strip()]
        plants.append(PlantResultResponse(
            id=pr.id, plant_id=pr.plant_id, status=pr.status,
            disease_labels=labels, confidence=pr.confidence,
            leaf_count=pr.leaf_count, diseased_leaf_count=pr.diseased_leaf_count,
            gps_lat=pr.gps_lat, gps_lon=pr.gps_lon, leaves=leaves,
        ))

    frames = [
        FrameResponse(
            frame_index=fr.frame_index, original_path=fr.original_path,
            annotated_path=fr.annotated_path, plant_count=fr.plant_count,
            leaf_count=fr.leaf_count,
        )
        for fr in sorted(flight.frames, key=lambda f: f.frame_index)
    ]

    return FlightDetailResponse(
        id=flight.id, video_filename=flight.video_filename,
        status=flight.status, current_stage=flight.current_stage or "",
        progress=flight.progress or 0, total_frames=flight.total_frames or 0,
        processed_frames=flight.processed_frames or 0,
        total_plants=flight.total_plants or 0,
        diseased_plants=flight.diseased_plants or 0,
        healthy_plants=flight.healthy_plants or 0,
        created_at=flight.created_at, completed_at=flight.completed_at,
        error_message=flight.error_message, plants=plants, frames=frames,
    )


@router.delete("/{flight_id}")
def delete_flight(
    flight_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    flight = db.query(Flight).filter(Flight.id == flight_id, Flight.user_id == user.id).first()
    if not flight:
        raise HTTPException(404, "Flight not found")
    db.delete(flight)
    db.commit()
    return {"deleted": True}


def _to_summary(f: Flight) -> FlightSummary:
    return FlightSummary(
        id=f.id, video_filename=f.video_filename, status=f.status,
        current_stage=f.current_stage or "", progress=f.progress or 0,
        total_frames=f.total_frames or 0, processed_frames=f.processed_frames or 0,
        total_plants=f.total_plants or 0, diseased_plants=f.diseased_plants or 0,
        healthy_plants=f.healthy_plants or 0, created_at=f.created_at,
        completed_at=f.completed_at,
    )
