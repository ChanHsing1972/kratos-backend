from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.hyperate import HyperateCurrentResponse
from app.services.auth import get_current_user
from app.services.hyperate import fetch_current_heart_rate, normalize_hyperate_id
from app.services.user_profile import get_profile_by_user_id

router = APIRouter()


@router.get("/hyperate/current", response_model=HyperateCurrentResponse)
def get_current_hyperate_heart_rate(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = get_profile_by_user_id(db, current_user.id)
    hyperate_id = normalize_hyperate_id(profile.hyperate_id if profile else None)
    if not hyperate_id:
        return {
            "bpm": None,
            "source": "hyperate",
            "recorded_at": datetime.utcnow(),
            "status": "unbound",
            "detail": "当前用户未绑定 HypeRate ID",
        }

    reading = fetch_current_heart_rate(hyperate_id)
    return {
        "bpm": reading.bpm,
        "source": reading.source,
        "recorded_at": reading.recorded_at,
        "status": reading.status,
        "detail": reading.detail,
    }
