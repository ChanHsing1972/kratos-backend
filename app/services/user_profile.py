from sqlalchemy.orm import Session

from app.models.user import User
from app.models.user_profile import UserProfile
from app.schemas.user_profile import UserProfileCreate, UserProfileUpdate


def get_profile_by_user_id(db: Session, user_id: int) -> UserProfile | None:
    return db.query(UserProfile).filter(UserProfile.user_id == user_id).first()


def create_profile_for_user(
    db: Session,
    user: User,
    profile_in: UserProfileCreate,
) -> UserProfile:
    profile = UserProfile(user_id=user.id, **profile_in.model_dump())
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def update_profile(
    db: Session,
    profile: UserProfile,
    profile_in: UserProfileUpdate,
) -> UserProfile:
    for field, value in profile_in.to_update_dict().items():
        setattr(profile, field, value)
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile
