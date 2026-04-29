"""Shared dependencies — current user extraction."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Header
from sqlalchemy.orm import Session

from app.auth import decode_token
from app.database import get_db
from app.models import User


def get_current_user(
    authorization: str = Header(default=""),
    db: Session = Depends(get_db),
) -> User:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(401, "Invalid token")
    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user:
        raise HTTPException(401, "User not found")
    return user
