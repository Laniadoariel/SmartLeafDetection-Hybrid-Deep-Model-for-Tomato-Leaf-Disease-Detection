"""SmartLeafDetection — FastAPI backend application."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import Base, engine, ensure_columns
from app.routes.auth_routes import router as auth_router
from app.routes.flight_routes import router as flight_router

# Create all tables, then add any newly-introduced columns to an existing DB.
Base.metadata.create_all(bind=engine)
ensure_columns()

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


@app.get("/api/model-info")
def model_info():
    """Report the active disease backend and the loaded classifier architecture.

    Lets you verify the WebApp is using the intended model (e.g. ResNet50) without
    processing a video. Reads the checkpoint metadata on demand.
    """
    from app import worker
    out = {
        "disease_backend": worker.DISEASE_BACKEND,
        "classifier_weights": worker.CLASSIFIER_WEIGHTS,
        "leaf_detector_weights": worker.LEAF_WEIGHTS,
        "disease_yolo_weights": worker.DISEASE_WEIGHTS,
        "classifier_arch": None,
    }
    if worker.CLASSIFIER_WEIGHTS:
        try:
            import torch
            ck = torch.load(worker.CLASSIFIER_WEIGHTS, map_location="cpu", weights_only=False)
            out["classifier_arch"] = ck.get("arch")
            out["classifier_classes"] = ck.get("classes")
            out["val_macro_f1"] = ck.get("val_macro_f1")
        except Exception as exc:  # pragma: no cover
            out["classifier_load_error"] = str(exc)
    return out
