from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.user_profile import (
    UserProfileCreate,
    UserProfileResponse,
    UserProfileUpdate,
)
from app.schemas.fitness_context import FitnessContextResponse
from app.services.auth import get_current_user
from app.services.fitness_context import load_fitness_context
from app.services.user_profile import (
    create_profile_for_user,
    get_profile_by_user_id,
    update_profile,
)

router = APIRouter()


@router.get("/context", response_model=FitnessContextResponse)
def get_my_fitness_context(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return load_fitness_context(db, current_user)


@router.get("/me", response_model=UserProfileResponse)
def get_my_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = get_profile_by_user_id(db, current_user.id)
    if not profile:
        raise HTTPException(status_code=404, detail="用户画像不存在")
    return profile


@router.post("/me", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
def create_my_profile(
    profile_in: UserProfileCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    existing_profile = get_profile_by_user_id(db, current_user.id)
    if existing_profile:
        raise HTTPException(status_code=400, detail="用户画像已存在")
    return create_profile_for_user(db, current_user, profile_in)


@router.put("/me", response_model=UserProfileResponse)
def update_my_profile(
    profile_in: UserProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = get_profile_by_user_id(db, current_user.id)
    if not profile:
        raise HTTPException(status_code=404, detail="用户画像不存在")
    return update_profile(db, profile, profile_in)


@router.patch("/me", response_model=UserProfileResponse)
def patch_my_profile(
    profile_in: UserProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = get_profile_by_user_id(db, current_user.id)
    if not profile:
        raise HTTPException(status_code=404, detail="用户画像不存在")
    return update_profile(db, profile, profile_in)
