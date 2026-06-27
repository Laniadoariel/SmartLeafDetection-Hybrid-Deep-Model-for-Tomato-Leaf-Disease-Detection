"""Pydantic schemas for API request/response validation."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    full_name: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str


class UserResponse(BaseModel):
    id: str
    username: str
    full_name: str


class FlightSummary(BaseModel):
    id: str
    video_filename: str
    status: str
    current_stage: str
    progress: float
    total_frames: int
    processed_frames: int
    total_video_frames: int = 0
    relevant_frames: int = 0
    total_detections: int = 0
    total_plants: int
    diseased_plants: int
    healthy_plants: int
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class LeafResultResponse(BaseModel):
    leaf_id: int
    frame_index: int = 0
    label: str
    confidence: float
    bbox: list[float]
    crop_path: Optional[str] = None

    class Config:
        from_attributes = True


class PlantResultResponse(BaseModel):
    id: str
    plant_id: int
    status: str
    disease_labels: list[str]
    confidence: float
    leaf_count: int
    diseased_leaf_count: int
    frames_seen: int = 0
    views_total: int = 0
    views_agreeing: int = 0
    weighted_decision: bool = False
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    leaves: list[LeafResultResponse] = []

    class Config:
        from_attributes = True


class FrameResponse(BaseModel):
    frame_index: int
    original_path: str
    annotated_path: Optional[str] = None
    plant_count: int
    leaf_count: int

    class Config:
        from_attributes = True


class FlightDetailResponse(BaseModel):
    id: str
    video_filename: str
    status: str
    current_stage: str
    progress: float
    total_frames: int
    processed_frames: int
    total_video_frames: int = 0
    relevant_frames: int = 0
    total_detections: int = 0
    total_plants: int
    diseased_plants: int
    healthy_plants: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    plants: list[PlantResultResponse] = []
    frames: list[FrameResponse] = []

    class Config:
        from_attributes = True


class StageUpdate(BaseModel):
    stage: str
    progress: float
    message: str = ""
