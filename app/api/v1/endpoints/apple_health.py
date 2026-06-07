from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.apple_health import (
    AppleHealthSyncRequest,
    AppleHealthSyncResponse,
    LatestAppleHealthSyncData,
    LatestAppleHealthSyncResponse,
)
from app.services.apple_health import (
    get_latest_apple_health_sync,
    save_apple_health_sync,
)
from app.services.auth import get_current_user

router = APIRouter()


@router.post("/sync", response_model=AppleHealthSyncResponse)
def sync_apple_health_data(
    payload: AppleHealthSyncRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    record = save_apple_health_sync(db, current_user, payload)
    return AppleHealthSyncResponse(
        success=True,
        message="Apple Health data synced successfully",
        data=record,
    )


@router.get("/latest", response_model=LatestAppleHealthSyncResponse)
def get_latest_apple_health_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    latest_sync = get_latest_apple_health_sync(db, current_user.id)
    if latest_sync is None:
        raise HTTPException(status_code=404, detail="No Apple Health data found for user")

    return LatestAppleHealthSyncResponse(
        success=True,
        message="Latest Apple Health data retrieved successfully",
        data=LatestAppleHealthSyncData(latest_sync=latest_sync),
    )
