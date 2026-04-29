"""SQLAlchemy ORM models."""
from __future__ import annotations

import datetime
import uuid

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=_uuid)
    username = Column(String(80), unique=True, nullable=False, index=True)
    full_name = Column(String(200), nullable=False)
    hashed_password = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    flights = relationship("Flight", back_populates="user")


class Flight(Base):
    __tablename__ = "flights"
    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    video_filename = Column(String(300), nullable=False)
    video_path = Column(String(500), nullable=False)
    status = Column(String(30), default="uploaded")  # uploaded/processing/completed/failed
    current_stage = Column(String(100), default="")
    progress = Column(Float, default=0.0)
    total_frames = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    total_plants = Column(Integer, default=0)
    diseased_plants = Column(Integer, default=0)
    healthy_plants = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    user = relationship("User", back_populates="flights")
    plant_results = relationship("PlantResult", back_populates="flight", cascade="all,delete-orphan")
    frames = relationship("FrameRecord", back_populates="flight", cascade="all,delete-orphan")


class PlantResult(Base):
    __tablename__ = "plant_results"
    id = Column(String(36), primary_key=True, default=_uuid)
    flight_id = Column(String(36), ForeignKey("flights.id"), nullable=False)
    plant_id = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False)  # healthy / diseased
    disease_labels = Column(Text, default="")  # comma-separated
    confidence = Column(Float, default=0.0)
    leaf_count = Column(Integer, default=0)
    diseased_leaf_count = Column(Integer, default=0)
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    gps_alt = Column(Float, nullable=True)
    evidence_json = Column(Text, default="{}")
    flight = relationship("Flight", back_populates="plant_results")
    leaf_results = relationship("LeafResult", back_populates="plant_result", cascade="all,delete-orphan")


class LeafResult(Base):
    __tablename__ = "leaf_results"
    id = Column(String(36), primary_key=True, default=_uuid)
    plant_result_id = Column(String(36), ForeignKey("plant_results.id"), nullable=False)
    leaf_id = Column(Integer, nullable=False)
    label = Column(String(100), nullable=False)
    confidence = Column(Float, default=0.0)
    bbox_x1 = Column(Float, default=0)
    bbox_y1 = Column(Float, default=0)
    bbox_x2 = Column(Float, default=0)
    bbox_y2 = Column(Float, default=0)
    crop_path = Column(String(500), nullable=True)
    plant_result = relationship("PlantResult", back_populates="leaf_results")


class FrameRecord(Base):
    __tablename__ = "frame_records"
    id = Column(String(36), primary_key=True, default=_uuid)
    flight_id = Column(String(36), ForeignKey("flights.id"), nullable=False)
    frame_index = Column(Integer, nullable=False)
    original_path = Column(String(500), nullable=False)
    annotated_path = Column(String(500), nullable=True)
    plant_count = Column(Integer, default=0)
    leaf_count = Column(Integer, default=0)
    flight = relationship("Flight", back_populates="frames")
