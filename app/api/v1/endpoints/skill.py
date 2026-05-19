from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.skill import (
    SkillBindingUpdate,
    SkillCreate,
    SkillResponse,
    SkillUpdate,
)
from app.services.auth import get_current_user
from app.services.skill import (
    create_skill,
    delete_skill,
    get_accessible_skill,
    get_editable_skill,
    is_skill_enabled_for_user,
    list_my_skills,
    list_skills_for_user,
    set_user_skill_enabled,
    update_skill,
)

router = APIRouter()


@router.get("", response_model=list[SkillResponse])
def list_skills(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_skills_for_user(db, current_user.id)


@router.get("/mine", response_model=list[SkillResponse])
def list_enabled_or_owned_skills(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_my_skills(db, current_user.id)


@router.post("", response_model=SkillResponse, status_code=status.HTTP_201_CREATED)
def create_custom_skill(
    skill_in: SkillCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return create_skill(db, current_user, skill_in)


@router.patch("/{skill_id}", response_model=SkillResponse)
def update_custom_skill(
    skill_id: int,
    skill_in: SkillUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    skill = get_editable_skill(db, skill_id, current_user.id)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill 不存在或不可编辑")
    return update_skill(
        db,
        skill,
        skill_in,
        enabled=is_skill_enabled_for_user(db, current_user.id, skill_id),
    )


@router.patch("/{skill_id}/binding", response_model=SkillResponse)
def update_skill_binding(
    skill_id: int,
    binding_in: SkillBindingUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    skill = get_accessible_skill(db, skill_id, current_user.id)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill 不存在")
    return set_user_skill_enabled(db, current_user.id, skill, binding_in.enabled)


@router.delete("/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_custom_skill(
    skill_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    skill = get_editable_skill(db, skill_id, current_user.id)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill 不存在或不可删除")
    delete_skill(db, skill)
