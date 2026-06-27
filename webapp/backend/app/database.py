"""Database setup and session management."""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./smartleaf.db",  # SQLite fallback when MySQL unavailable
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def ensure_columns() -> None:
    """Lightweight, idempotent migration for the dev SQLite DB.

    ``Base.metadata.create_all`` creates missing *tables* but never alters
    existing ones, so newly-added columns would be missing on an old
    ``smartleaf.db``. This adds any missing columns via ``ALTER TABLE ADD
    COLUMN`` (SQLite-supported), preserving existing rows. No-op on non-SQLite.
    """
    if engine.dialect.name != "sqlite":
        return
    wanted = {
        "flights": [
            ("total_video_frames", "INTEGER", "0"),
            ("relevant_frames", "INTEGER", "0"),
            ("total_detections", "INTEGER", "0"),
        ],
        "plant_results": [
            ("views_total", "INTEGER", "0"),
            ("views_agreeing", "INTEGER", "0"),
            ("weighted_decision", "INTEGER", "0"),
        ],
        "leaf_results": [
            ("frame_index", "INTEGER", "0"),
        ],
    }
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in wanted.items():
            if not insp.has_table(table):
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, sqltype, default in cols:
                if name not in existing:
                    conn.execute(text(
                        f"ALTER TABLE {table} ADD COLUMN {name} {sqltype} DEFAULT {default}"
                    ))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
